"""Configuration management using pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Svend application settings.

    Loaded from environment variables with SVEND_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="SVEND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    env: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = True
    secret_key: str = "change-me-in-production"

    # Database (SQLite for dev, Postgres for prod)
    # Postgres: postgresql://user:pass@localhost:5432/svend
    database_url: str = Field(
        default="sqlite:///./db.sqlite3",
        description="Database connection string",
    )

    # Stripe billing
    stripe_secret_key: str = Field(
        default="",
        description="Stripe secret key (sk_live_... or sk_test_...)",
    )
    stripe_webhook_secret: str = Field(
        default="",
        description="Stripe webhook signing secret (whsec_...)",
    )
    stripe_price_id_pro: str = Field(
        default="",
        description="Stripe Price ID for Pro tier ($5/month)",
    )

    # Kjerne pipeline
    kjerne_path: Path = Field(
        default=Path(__file__).resolve().parent.parent.parent / "k" / "kjerne",
        description="Path to kjerne ensemble models",
    )
    device: str = "cuda"

    # API
    allowed_hosts: str = "localhost,127.0.0.1,0.0.0.0"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Rate limiting
    rate_limit_per_minute: int = 20

    # Alpha access control (set to false for open registration)
    require_invite: bool = True

    # Pipeline selection (Synara = alpha-ready with 93% accuracy, MoE = requires training)
    use_synara: bool = Field(
        default=True,
        description="Use Synara pipeline (True) or MoE pipeline (False)",
    )

    # Language model settings
    use_open_source_lm: bool = Field(
        default=False,
        description="Use open source LM (Qwen) instead of trained LM",
    )
    open_source_lm_model: str = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        description="HuggingFace model ID for open source LM",
    )

    # Coder pipeline (Qwen-Coder for computation tasks)
    enable_coder: bool = Field(
        default=True,
        description="Enable Qwen-Coder for computation/visualization tasks",
    )
    coder_model: str = Field(
        default="Qwen/Qwen2.5-Coder-14B-Instruct",
        description="HuggingFace model ID for code generation",
    )

    # Flywheel / Opus escalation
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Opus escalation (flywheel)",
    )
    flywheel_confidence_threshold: float = Field(
        default=0.7,
        description="Below this confidence, consider Opus escalation",
    )
    flywheel_auto_escalate_threshold: float = Field(
        default=0.4,
        description="Below this confidence, auto-escalate to Opus",
    )

    # Email settings (SMTP)
    email_host: str = Field(
        default="smtp.resend.com",
        description="SMTP server host",
    )
    email_port: int = Field(
        default=587,
        description="SMTP server port",
    )
    email_host_user: str = Field(
        default="resend",
        description="SMTP username",
    )
    email_host_password: str = Field(
        default="",
        description="SMTP password or API key",
    )
    email_use_tls: bool = Field(
        default=True,
        description="Use TLS for SMTP",
    )
    email_from: str = Field(
        default="Svend <hello@svend.ai>",
        description="Default from email address",
    )

    def get_allowed_hosts(self) -> list[str]:
        return [h.strip() for h in self.allowed_hosts.split(",")]

    def get_cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
