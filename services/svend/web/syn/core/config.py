"""
Deployment Configuration (CONFIG-001)
=====================================

Pydantic BaseSettings implementation for configuration management.

Standard:     CONFIG-001 §6 (Settings Schema)
Compliance:   NIST SP 800-53 CM-2, ISO 27001 A.8.9
Location:     syn/core/config.py
Version:      1.0.0

Features:
- Environment variable loading with validation
- Feature flag registry
- Deployment profile support
- Secrets classification
"""

import os
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# ENUMS (CONFIG-001 §5, §7)
# =============================================================================


class DeploymentProfile(str, Enum):
    """Deployment profile per CONFIG-001 §7."""

    LOCAL = "local"
    DOCKER = "docker"
    STAGING = "staging"
    PRODUCTION = "production"
    RENDER = "render"


class FeatureFlagState(str, Enum):
    """Feature flag lifecycle state per CONFIG-001 §5.2."""

    DISABLED = "disabled"
    BETA = "beta"
    GA = "ga"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"


class SecretClassification(str, Enum):
    """Secret classification per CONFIG-001 §9.1."""

    CRITICAL = "critical"  # Vault only: DJANGO_SECRET_KEY, JWT_SECRET
    SENSITIVE = "sensitive"  # Vault or platform: API keys, passwords
    INTERNAL = "internal"  # Env or config: service URLs
    PUBLIC = "public"  # Code defaults: LOG_LEVEL


class LogLevel(str, Enum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log format enumeration."""

    JSON = "json"
    TEXT = "text"


# =============================================================================
# FEATURE FLAGS (CONFIG-001 §5)
# =============================================================================


class FeatureFlags(BaseSettings):
    """
    Feature flag registry per CONFIG-001 §5.

    All feature flags should be explicitly declared here with:
    - Default value
    - State (GA, beta, deprecated)
    - Owner team
    """

    model_config = SettingsConfigDict(
        env_prefix="FEATURE_",
        extra="ignore",
    )

    # Core features
    enable_soft_delete_cascade: bool = Field(
        default=True,
        description="Allow cascade soft delete operations (ORG-002 §4.1)",
    )
    enable_membership_reassignment: bool = Field(
        default=True,
        description="Auto-reassign memberships on domain delete",
    )
    enable_cross_domain_audit: bool = Field(
        default=True,
        description="Log cross-domain access patterns (ORG-001 §12.11)",
    )

    # Authentication features
    enable_oidc: bool = Field(
        default=False,
        alias="OIDC_ENABLED",
        description="Enable OIDC SSO authentication",
    )
    enable_api_keys: bool = Field(
        default=False,
        alias="API_KEYS_ENABLED",
        description="Enable API key authentication",
    )

    # Security features
    enable_vault_secrets: bool = Field(
        default=False,
        alias="VAULT_ENABLED",
        description="Load secrets from HashiCorp Vault",
    )

    # Module features
    enable_cognition_engine: bool = Field(
        default=True,
        description="Enable cognitive reasoning engine (COG-001)",
    )
    enable_erm_module: bool = Field(
        default=True,
        description="Enable External Relationship Management (ERM-001)",
    )
    enable_governance_preflight: bool = Field(
        default=True,
        description="Enable governance preflight checks (GOV-001)",
    )


# =============================================================================
# DATABASE SETTINGS (CONFIG-001 §4.2)
# =============================================================================


class DatabaseSettings(BaseSettings):
    """Database configuration per CONFIG-001 §4.2."""

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        extra="ignore",
    )

    engine: str = Field(
        default="django.db.backends.postgresql",
        description="Database backend",
    )
    name: str = Field(
        description="Database name",
    )
    user: str = Field(
        description="Database username",
    )
    password: SecretStr = Field(
        description="Database password (CRITICAL)",
    )
    host: str = Field(
        default="localhost",
        description="Database hostname",
    )
    port: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="Database port",
    )

    # Connection URL alternative
    url: Optional[str] = Field(
        default=None,
        alias="DATABASE_URL",
        description="Full connection URL (alternative to individual fields)",
    )

    # Connection pooling
    conn_max_age: int = Field(
        default=60,
        ge=0,
        description="Connection max age in seconds",
    )

    @property
    def as_django_config(self) -> Dict[str, Any]:
        """Convert to Django DATABASES config format."""
        if self.url:
            import dj_database_url

            return dj_database_url.parse(self.url)

        return {
            "ENGINE": self.engine,
            "NAME": self.name,
            "USER": self.user,
            "PASSWORD": self.password.get_secret_value(),
            "HOST": self.host,
            "PORT": self.port,
            "CONN_MAX_AGE": self.conn_max_age,
        }


# =============================================================================
# REDIS SETTINGS (CONFIG-001 §4.2)
# =============================================================================


class RedisSettings(BaseSettings):
    """Redis configuration per CONFIG-001 §4.2."""

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        extra="ignore",
    )

    url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )


# =============================================================================
# CELERY SETTINGS (CONFIG-001 §4.2)
# =============================================================================


class CelerySettings(BaseSettings):
    """Celery configuration per CONFIG-001 §4.2."""

    model_config = SettingsConfigDict(
        env_prefix="CELERY_",
        extra="ignore",
    )

    broker_url: str = Field(
        description="Celery message broker URL",
    )
    result_backend: str = Field(
        description="Celery result backend URL",
    )
    worker_concurrency: int = Field(
        default=2,
        ge=1,
        le=32,
        description="Worker process count",
    )
    task_acks_late: bool = Field(
        default=True,
        description="Acknowledge after completion",
    )
    task_reject_on_worker_lost: bool = Field(
        default=True,
        description="Reject tasks when worker lost",
    )


# =============================================================================
# AUTHENTICATION SETTINGS (CONFIG-001 §4.3)
# =============================================================================


class AuthenticationSettings(BaseSettings):
    """Authentication configuration per CONFIG-001 §4.3."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
    )

    jwt_secret: SecretStr = Field(
        alias="JWT_SECRET",
        description="JWT signing secret (CRITICAL)",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        alias="JWT_ALGORITHM",
        description="JWT signing algorithm",
    )
    jwt_expiration_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        alias="JWT_EXPIRATION_SECONDS",
        description="Token lifetime in seconds",
    )
    require_auth: bool = Field(
        default=True,
        alias="REQUIRE_AUTH",
        description="Require authentication",
    )

    @field_validator("jwt_algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        allowed = {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512"}
        if v not in allowed:
            raise ValueError(f"JWT algorithm must be one of {allowed}")
        return v


# =============================================================================
# OBSERVABILITY SETTINGS (CONFIG-001 §4.2)
# =============================================================================


class ObservabilitySettings(BaseSettings):
    """Observability configuration per CONFIG-001 §4.2."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
    )

    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        alias="LOG_LEVEL",
        description="Logging level",
    )
    log_format: LogFormat = Field(
        default=LogFormat.JSON,
        alias="LOG_FORMAT",
        description="Log format",
    )

    # OpenTelemetry
    otel_endpoint: Optional[str] = Field(
        default=None,
        alias="OTEL_EXPORTER_OTLP_ENDPOINT",
        description="OpenTelemetry collector endpoint",
    )
    otel_service_name: str = Field(
        default="synara-core",
        alias="OTEL_SERVICE_NAME",
        description="Service name for traces",
    )

    # OPA
    opa_url: Optional[str] = Field(
        default=None,
        alias="OPA_URL",
        description="Open Policy Agent URL",
    )


# =============================================================================
# SECURITY SETTINGS (CONFIG-001 §4.3)
# =============================================================================


class SecuritySettings(BaseSettings):
    """Security configuration per CONFIG-001 §4.3."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
    )

    csrf_cookie_secure: bool = Field(
        default=False,
        alias="CSRF_COOKIE_SECURE",
        description="CSRF cookie over HTTPS only",
    )
    session_cookie_secure: bool = Field(
        default=False,
        alias="SESSION_COOKIE_SECURE",
        description="Session cookie over HTTPS only",
    )
    secure_ssl_redirect: bool = Field(
        default=False,
        alias="SECURE_SSL_REDIRECT",
        description="Redirect HTTP to HTTPS",
    )
    secure_content_type_nosniff: bool = Field(
        default=True,
        alias="SECURE_CONTENT_TYPE_NOSNIFF",
        description="Prevent MIME sniffing",
    )
    x_frame_options: str = Field(
        default="SAMEORIGIN",
        alias="X_FRAME_OPTIONS",
        description="Clickjacking protection",
    )


# =============================================================================
# MAIN SETTINGS (CONFIG-001 §6)
# =============================================================================


class SynaraSettings(BaseSettings):
    """
    Main Synara configuration per CONFIG-001 §6.

    Standard: CONFIG-001 §6.1
    Compliance: NIST SP 800-53 CM-2, CM-3

    This is the root configuration class that aggregates all settings.
    Use get_settings() to get a cached instance.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Django core
    django_secret_key: SecretStr = Field(
        alias="DJANGO_SECRET_KEY",
        description="Cryptographic signing key (CRITICAL)",
    )
    django_debug: bool = Field(
        default=False,
        alias="DJANGO_DEBUG",
        description="Debug mode (NEVER True in production)",
    )
    django_allowed_hosts: str = Field(
        default="localhost,127.0.0.1",
        alias="DJANGO_ALLOWED_HOSTS",
        description="Comma-separated allowed hostnames",
    )
    django_settings_module: str = Field(
        default="synara_core.settings",
        alias="DJANGO_SETTINGS_MODULE",
        description="Python settings module path",
    )

    # Deployment
    deployment_profile: DeploymentProfile = Field(
        default=DeploymentProfile.LOCAL,
        alias="DEPLOYMENT_PROFILE",
        description="Deployment profile",
    )

    # Sub-configurations (loaded from environment)
    # Note: These are lazy-loaded to avoid validation errors when not all vars are set

    @property
    def allowed_hosts_list(self) -> List[str]:
        """Parse allowed hosts as list."""
        return [h.strip() for h in self.django_allowed_hosts.split(",") if h.strip()]

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.deployment_profile == DeploymentProfile.PRODUCTION

    @property
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.django_debug

    @model_validator(mode="after")
    def validate_production_settings(self) -> "SynaraSettings":
        """Validate production security settings per CONFIG-001 §10."""
        if self.is_production:
            if self.django_debug:
                raise ValueError("DEBUG must be False in production")
            if "*" in self.django_allowed_hosts:
                raise ValueError("ALLOWED_HOSTS cannot be * in production")
        return self

    def get_feature_flags(self) -> FeatureFlags:
        """Get feature flags configuration."""
        return FeatureFlags()

    def get_observability(self) -> ObservabilitySettings:
        """Get observability configuration."""
        return ObservabilitySettings()

    def get_security(self) -> SecuritySettings:
        """Get security configuration."""
        return SecuritySettings()


# =============================================================================
# SETTINGS ACCESS (CONFIG-001 §6.2)
# =============================================================================


@lru_cache()
def get_settings() -> SynaraSettings:
    """
    Get cached settings instance.

    Standard: CONFIG-001 §6.2 (Configuration Validation at Startup)

    Returns:
        SynaraSettings instance (cached)

    Raises:
        ValidationError: If configuration is invalid
    """
    return SynaraSettings()


def validate_settings() -> None:
    """
    Validate all settings at startup.

    Standard: CONFIG-001 §6.2

    Raises:
        ValidationError: If any configuration is invalid
    """
    # This will raise ValidationError if settings are invalid
    settings = get_settings()

    # Validate feature flags
    _ = settings.get_feature_flags()

    # Validate observability
    _ = settings.get_observability()

    # Validate security
    _ = settings.get_security()


# =============================================================================
# ANTI-PATTERN CHECKS (CONFIG-001 §10)
# =============================================================================


def check_anti_patterns() -> List[str]:
    """
    Check for configuration anti-patterns per CONFIG-001 §10.

    Returns:
        List of warnings/errors
    """
    warnings = []
    settings = get_settings()

    # CONFIG-001-AP-003: Debug in Production
    if settings.is_production and settings.django_debug:
        warnings.append("CONFIG-001-AP-003: DEBUG=True in production")

    # CONFIG-001-AP-003: ALLOWED_HOSTS=* in production
    if settings.is_production and "*" in settings.allowed_hosts_list:
        warnings.append("CONFIG-001-AP-003: ALLOWED_HOSTS=* in production")

    # Check security settings in production
    if settings.is_production:
        security = settings.get_security()
        if not security.csrf_cookie_secure:
            warnings.append("CSRF_COOKIE_SECURE should be True in production")
        if not security.session_cookie_secure:
            warnings.append("SESSION_COOKIE_SECURE should be True in production")
        if not security.secure_ssl_redirect:
            warnings.append("SECURE_SSL_REDIRECT should be True in production")

    return warnings
