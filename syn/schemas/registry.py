"""
Schema Registry (SCONF-001 §10)
===============================

Schema registry for loading, caching, and managing schemas.

Standard:     SCONF-001 §10 (Registry Configuration)
Compliance:   VAL-002 §4.4, SCHEMA-001 §6
Location:     syn/schemas/registry.py
Version:      1.0.0

Features:
- Schema registration and lookup
- Redis-based caching
- Performance-optimized retrieval
- Bootstrap schema loading
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from syn.schemas.types import (
    QMS_REQUIRED_FIELDS,
    QMS_SCHEMA_PATTERNS,
    QMS_SCHEMA_URIS,
    QMSSchemaId,
    SchemaStatus,
    SchemaType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# REGISTRY CONFIGURATION (SCONF-001 §10.2)
# =============================================================================


@dataclass
class SchemaRegistryConfig:
    """
    Schema registry configuration per SCONF-001 §10.2.
    """

    # Cache settings
    cache_backend: str = "redis"
    cache_ttl_seconds: int = 300
    cache_prefix: str = "schema:"

    # Performance targets
    p95_cached_ms: int = 2
    p95_uncached_ms: int = 10

    # Version retention
    max_versions_per_schema: int = 10
    auto_retire_after_days: int = 365


# Default configuration
DEFAULT_CONFIG = SchemaRegistryConfig()


# =============================================================================
# SCHEMA DEFINITION (SCONF-001 §6)
# =============================================================================


@dataclass
class SchemaField:
    """
    Schema field definition.
    """

    name: str
    field_type: str
    required: bool = False
    default: Any = None
    description: Optional[str] = None
    constraints: Optional[Dict] = None
    governance_tags: List[str] = field(default_factory=list)


@dataclass
class SchemaDefinition:
    """
    Complete schema definition per SCONF-001 §6.
    """

    id: str
    uri: str
    title: str
    schema_type: SchemaType
    version: str
    status: SchemaStatus = SchemaStatus.DRAFT
    pattern: Optional[str] = None  # ID pattern (e.g., "CAPA-YYYY-NNNN")
    fields: List[SchemaField] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    checksum: Optional[str] = None

    def compute_checksum(self) -> str:
        """Compute SHA-256 checksum of schema definition."""
        content = json.dumps(
            {
                "id": self.id,
                "uri": self.uri,
                "version": self.version,
                "fields": [f.name for f in self.fields],
                "required_fields": self.required_fields,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_json_schema(self) -> Dict:
        """Convert to JSON Schema format."""
        properties = {}
        required = []

        for f in self.fields:
            prop = {"type": f.field_type}
            if f.description:
                prop["description"] = f.description
            if f.constraints:
                prop.update(f.constraints)
            properties[f.name] = prop
            if f.required:
                required.append(f.name)

        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": self.uri,
            "title": self.title,
            "type": "object",
            "properties": properties,
            "required": required or self.required_fields,
        }


# =============================================================================
# SCHEMA REGISTRY (SCONF-001 §10.3)
# =============================================================================


class SchemaRegistry:
    """
    Schema registry per SCONF-001 §10.3.

    Provides:
    - Schema registration and lookup
    - Version management
    - Caching for performance
    - Bootstrap loading
    """

    def __init__(self, config: Optional[SchemaRegistryConfig] = None):
        """
        Initialize registry.

        Args:
            config: Registry configuration (uses default if not provided)
        """
        self.config = config or DEFAULT_CONFIG
        self._schemas: Dict[str, SchemaDefinition] = {}
        self._by_id: Dict[str, List[str]] = {}  # id -> list of URIs (versions)
        self._cache_hits = 0
        self._cache_misses = 0

    def register(self, schema: SchemaDefinition) -> None:
        """
        Register a schema definition.

        Args:
            schema: Schema to register
        """
        # Compute checksum
        schema.checksum = schema.compute_checksum()

        # Store by URI
        self._schemas[schema.uri] = schema

        # Index by ID
        if schema.id not in self._by_id:
            self._by_id[schema.id] = []
        if schema.uri not in self._by_id[schema.id]:
            self._by_id[schema.id].append(schema.uri)

        # Enforce version retention
        self._enforce_retention(schema.id)

        logger.info(f"Registered schema: {schema.uri}")

    def _enforce_retention(self, schema_id: str) -> None:
        """Enforce max version retention per SCONF-001 §10.2."""
        uris = self._by_id.get(schema_id, [])
        if len(uris) > self.config.max_versions_per_schema:
            # Remove oldest versions
            to_remove = uris[: -self.config.max_versions_per_schema]
            for uri in to_remove:
                if uri in self._schemas:
                    del self._schemas[uri]
                self._by_id[schema_id].remove(uri)
                logger.info(f"Retired old schema version: {uri}")

    def get(self, uri: str) -> Optional[SchemaDefinition]:
        """
        Get schema by URI.

        Args:
            uri: Schema URI

        Returns:
            SchemaDefinition or None
        """
        schema = self._schemas.get(uri)
        if schema:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        return schema

    def get_latest(self, schema_id: str) -> Optional[SchemaDefinition]:
        """
        Get latest version of a schema by ID.

        Args:
            schema_id: Schema ID (e.g., "qms.capa")

        Returns:
            Latest SchemaDefinition or None
        """
        uris = self._by_id.get(schema_id, [])
        if not uris:
            return None
        return self._schemas.get(uris[-1])

    def get_all_versions(self, schema_id: str) -> List[SchemaDefinition]:
        """
        Get all versions of a schema.

        Args:
            schema_id: Schema ID

        Returns:
            List of SchemaDefinitions (oldest to newest)
        """
        uris = self._by_id.get(schema_id, [])
        return [self._schemas[uri] for uri in uris if uri in self._schemas]

    def list_schemas(
        self,
        schema_type: Optional[SchemaType] = None,
        status: Optional[SchemaStatus] = None,
    ) -> List[SchemaDefinition]:
        """
        List schemas with optional filtering.

        Args:
            schema_type: Filter by type
            status: Filter by status

        Returns:
            List of matching schemas
        """
        schemas = list(self._schemas.values())

        if schema_type:
            schemas = [s for s in schemas if s.schema_type == schema_type]
        if status:
            schemas = [s for s in schemas if s.status == status]

        return schemas

    def validate_data(self, uri: str, data: Dict) -> List[str]:
        """
        Validate data against a schema.

        Args:
            uri: Schema URI
            data: Data to validate

        Returns:
            List of validation errors (empty if valid)
        """
        schema = self.get(uri)
        if not schema:
            return [f"Schema not found: {uri}"]

        errors = []

        # Check required fields
        for field_name in schema.required_fields:
            if field_name not in data or data[field_name] is None:
                errors.append(f"Missing required field: {field_name}")

        return errors

    @property
    def stats(self) -> Dict:
        """Get registry statistics."""
        return {
            "total_schemas": len(self._schemas),
            "unique_ids": len(self._by_id),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0
            ),
        }


# =============================================================================
# SCHEMA LOADER (SCONF-001 §10.3)
# =============================================================================


class SchemaLoader:
    """
    Schema loader for bootstrap and file-based loading.

    Standard: SCONF-001 §10.3
    """

    def __init__(self, registry: SchemaRegistry):
        """
        Initialize loader.

        Args:
            registry: Target registry
        """
        self.registry = registry

    def bootstrap(self) -> int:
        """
        Bootstrap QMS and common schemas per SCONF-001 §12.

        Returns:
            Number of schemas loaded
        """
        loaded = 0

        # Load QMS schemas
        for schema_id in QMSSchemaId:
            schema = self._create_qms_schema(schema_id)
            self.registry.register(schema)
            loaded += 1

        logger.info(f"Bootstrap complete: {loaded} schemas loaded")
        return loaded

    def _create_qms_schema(self, schema_id: QMSSchemaId) -> SchemaDefinition:
        """Create QMS schema definition from configuration."""
        uri = QMS_SCHEMA_URIS.get(schema_id, f"sconf://qms/{schema_id.value}/1.0.0")
        pattern = QMS_SCHEMA_PATTERNS.get(schema_id)
        required = QMS_REQUIRED_FIELDS.get(schema_id, [])

        # Create fields from required list
        fields = [SchemaField(name=f, field_type="string", required=True) for f in required]

        return SchemaDefinition(
            id=schema_id.value,
            uri=uri,
            title=schema_id.name.replace("_", " ").title(),
            schema_type=SchemaType.FORM_QMS,
            version="1.0.0",
            status=SchemaStatus.ACTIVE,
            pattern=pattern,
            fields=fields,
            required_fields=required,
        )

    def load_from_file(self, filepath: str) -> Optional[SchemaDefinition]:
        """
        Load schema from JSON file.

        Args:
            filepath: Path to JSON schema file

        Returns:
            SchemaDefinition or None on error
        """
        try:
            import json

            with open(filepath, "r") as f:
                data = json.load(f)

            schema = SchemaDefinition(
                id=data.get("id", ""),
                uri=data.get("$id", data.get("uri", "")),
                title=data.get("title", ""),
                schema_type=SchemaType(data.get("type", "form.custom")),
                version=data.get("version", "1.0.0"),
                required_fields=data.get("required", []),
            )

            self.registry.register(schema)
            return schema

        except Exception as e:
            logger.error(f"Failed to load schema from {filepath}: {e}")
            return None


# =============================================================================
# SINGLETON REGISTRY
# =============================================================================


_registry: Optional[SchemaRegistry] = None


def get_registry() -> SchemaRegistry:
    """Get global schema registry instance."""
    global _registry
    if _registry is None:
        _registry = SchemaRegistry()
    return _registry


def bootstrap_schemas() -> int:
    """Bootstrap schemas into global registry."""
    registry = get_registry()
    loader = SchemaLoader(registry)
    return loader.bootstrap()
