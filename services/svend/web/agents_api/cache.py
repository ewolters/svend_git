"""Database-backed session cache with Tempora TTL cleanup.

Replaces in-memory caches (synara instances, model cache, etc.) with
persistent database storage. Tempora handles TTL cleanup via scheduled tasks.

Usage:
    from agents_api.cache import SessionCache

    # Store a value
    SessionCache.set("synara:user123", synara_instance, ttl_seconds=3600)

    # Get a value
    instance = SessionCache.get("synara:user123")

    # Delete a value
    SessionCache.delete("synara:user123")

    # Get or create with factory
    instance = SessionCache.get_or_create(
        "synara:user123",
        factory=lambda: create_synara_instance(),
        ttl_seconds=3600
    )
"""

import json
import logging
import pickle
from datetime import timedelta
from typing import Any, Callable, Optional

from django.db import models
from django.utils import timezone

from .models import CacheEntry

logger = logging.getLogger(__name__)


class SessionCache:
    """High-level cache interface.

    Provides a simple key-value interface backed by CacheEntry model.
    """

    @classmethod
    def set(
        cls,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        namespace: str = "",
        user_id: Optional[int] = None,
        value_type: str = "pickle",
    ) -> bool:
        """Store a value in the cache.

        Args:
            key: Cache key
            value: Value to store (will be pickled)
            ttl_seconds: Time-to-live in seconds (None = no expiration)
            namespace: Optional namespace for grouping (e.g., "synara", "model")
            user_id: Optional user ID for user-scoped caching
            value_type: Serialization type ("pickle" or "json")

        Returns:
            True if successful
        """
        try:
            # Serialize value
            if value_type == "json":
                serialized = json.dumps(value).encode("utf-8")
            else:
                serialized = pickle.dumps(value)

            # Calculate expiration
            expires_at = None
            if ttl_seconds:
                expires_at = timezone.now() + timedelta(seconds=ttl_seconds)

            # Upsert
            entry, created = CacheEntry.objects.update_or_create(
                key=key,
                defaults={
                    "value": serialized,
                    "value_type": value_type,
                    "expires_at": expires_at,
                    "namespace": namespace,
                    "user_id": user_id,
                },
            )
            return True

        except Exception as e:
            logger.error(f"Cache set failed for {key}: {e}")
            return False

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a value from the cache.

        Args:
            key: Cache key
            default: Value to return if key not found or expired

        Returns:
            Cached value or default
        """
        try:
            entry = CacheEntry.objects.get(key=key)

            # Check expiration
            if entry.is_expired:
                entry.delete()
                return default

            # Update hit count
            entry.hit_count += 1
            entry.save(update_fields=["hit_count", "last_accessed"])

            # Deserialize
            if entry.value_type == "json":
                return json.loads(entry.value.decode("utf-8"))
            else:
                return pickle.loads(entry.value)

        except CacheEntry.DoesNotExist:
            return default
        except Exception as e:
            logger.error(f"Cache get failed for {key}: {e}")
            return default

    @classmethod
    def delete(cls, key: str) -> bool:
        """Delete a value from the cache."""
        try:
            deleted, _ = CacheEntry.objects.filter(key=key).delete()
            return deleted > 0
        except Exception as e:
            logger.error(f"Cache delete failed for {key}: {e}")
            return False

    @classmethod
    def exists(cls, key: str) -> bool:
        """Check if a key exists and is not expired."""
        try:
            entry = CacheEntry.objects.get(key=key)
            if entry.is_expired:
                entry.delete()
                return False
            return True
        except CacheEntry.DoesNotExist:
            return False

    @classmethod
    def get_or_create(
        cls,
        key: str,
        factory: Callable[[], Any],
        ttl_seconds: Optional[int] = None,
        namespace: str = "",
        user_id: Optional[int] = None,
    ) -> Any:
        """Get a value or create it using factory function.

        Args:
            key: Cache key
            factory: Function to create value if not cached
            ttl_seconds: TTL for newly created values
            namespace: Namespace for grouping
            user_id: User ID for user-scoped caching

        Returns:
            Cached or newly created value
        """
        value = cls.get(key)
        if value is not None:
            return value

        # Create new value
        value = factory()
        cls.set(key, value, ttl_seconds=ttl_seconds, namespace=namespace, user_id=user_id)
        return value

    @classmethod
    def clear_namespace(cls, namespace: str) -> int:
        """Clear all entries in a namespace."""
        deleted, _ = CacheEntry.objects.filter(namespace=namespace).delete()
        return deleted

    @classmethod
    def clear_user(cls, user_id: int) -> int:
        """Clear all entries for a user."""
        deleted, _ = CacheEntry.objects.filter(user_id=user_id).delete()
        return deleted

    @classmethod
    def clear_expired(cls) -> int:
        """Clear all expired entries. Called by Tempora cleanup task."""
        deleted, _ = CacheEntry.objects.filter(
            expires_at__isnull=False,
            expires_at__lt=timezone.now(),
        ).delete()
        return deleted

    @classmethod
    def stats(cls) -> dict:
        """Get cache statistics."""
        total = CacheEntry.objects.count()
        expired = CacheEntry.objects.filter(
            expires_at__isnull=False,
            expires_at__lt=timezone.now(),
        ).count()

        by_namespace = {}
        for entry in CacheEntry.objects.values("namespace").annotate(
            count=models.Count("id")
        ):
            ns = entry["namespace"] or "(default)"
            by_namespace[ns] = entry["count"]

        return {
            "total_entries": total,
            "expired_entries": expired,
            "by_namespace": by_namespace,
        }


# =============================================================================
# Tempora Integration - Cache cleanup task
# =============================================================================

def register_cache_cleanup_task():
    """Register the cache cleanup task with Tempora.

    Call this during app startup to schedule periodic cache cleanup.
    """
    try:
        from tempora.scheduler import Scheduler

        scheduler = Scheduler()

        # Schedule cleanup every 5 minutes
        scheduler.schedule_recurring(
            task_name="cache.cleanup",
            handler=cleanup_expired_cache,
            interval_minutes=5,
            description="Clean up expired session cache entries",
        )
        logger.info("Cache cleanup task registered with Tempora")

    except ImportError:
        logger.warning("Tempora not available, cache cleanup disabled")
    except Exception as e:
        logger.error(f"Failed to register cache cleanup task: {e}")


def cleanup_expired_cache():
    """Tempora task handler for cache cleanup."""
    deleted = SessionCache.clear_expired()
    logger.info(f"Cache cleanup: removed {deleted} expired entries")
    return {"deleted": deleted}


# =============================================================================
# Convenience functions for common cache patterns
# =============================================================================

class SynaraCache:
    """Cache manager for Synara instances."""

    NAMESPACE = "synara"
    DEFAULT_TTL = 3600  # 1 hour

    @classmethod
    def get(cls, user_id: int) -> Any:
        key = f"{cls.NAMESPACE}:user:{user_id}"
        return SessionCache.get(key)

    @classmethod
    def set(cls, user_id: int, instance: Any, ttl_seconds: int = None) -> bool:
        key = f"{cls.NAMESPACE}:user:{user_id}"
        return SessionCache.set(
            key,
            instance,
            ttl_seconds=ttl_seconds or cls.DEFAULT_TTL,
            namespace=cls.NAMESPACE,
            user_id=user_id,
        )

    @classmethod
    def delete(cls, user_id: int) -> bool:
        key = f"{cls.NAMESPACE}:user:{user_id}"
        return SessionCache.delete(key)


class ModelCache:
    """Cache manager for ML model results."""

    NAMESPACE = "model"
    DEFAULT_TTL = 1800  # 30 minutes

    @classmethod
    def get(cls, result_id: str) -> Any:
        key = f"{cls.NAMESPACE}:result:{result_id}"
        return SessionCache.get(key)

    @classmethod
    def set(cls, result_id: str, result: Any, ttl_seconds: int = None) -> bool:
        key = f"{cls.NAMESPACE}:result:{result_id}"
        return SessionCache.set(
            key,
            result,
            ttl_seconds=ttl_seconds or cls.DEFAULT_TTL,
            namespace=cls.NAMESPACE,
        )

    @classmethod
    def delete(cls, result_id: str) -> bool:
        key = f"{cls.NAMESPACE}:result:{result_id}"
        return SessionCache.delete(key)
