"""Django settings for Svend."""

import hashlib
import importlib
import sys
from pathlib import Path

import dj_database_url

from svend_config.config import get_settings

# Build paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Load pydantic settings
config = get_settings()

# Security
SECRET_KEY = config.secret_key
DEBUG = config.debug
ALLOWED_HOSTS = config.get_allowed_hosts()

# Application definition
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.sitemaps",
    # Third-party
    "rest_framework",
    "corsheaders",
    # Local apps
    "core",  # Foundation: tenants, projects, hypotheses, knowledge graph
    "accounts",
    "chat",
    "api",
    "forge",
    "files",
    "qms_core",  # QMS Core: permissions, Site (Phase 4). Object 271 extraction target.
    "agents_api.apps.AgentsApiConfig",
    "workbench",
    "notifications",
    "safety",
    "loop",  # LOOP-001: Closed-loop operating model (Signal, Commitment, ModeTransition, QMSPolicy)
    "graph",  # GRAPH-001: Unified Knowledge Graph and Process Model
    # ---- Synara Infrastructure (OS layer) ----
    "syn.core.apps.CoreConfig",  # label="syn_core"
    "syn.audit.apps.AuditConfig",  # label="audit"
    "syn.log.apps.LogConfig",  # label="syn_log"
    "syn.sched.apps.SchedConfig",  # label="sched"
    # NOTE: syn.api and syn.synara are NOT registered (no models).
    # syn.err is pure Python, not a Django app.
]

MIDDLEWARE = [
    # Django security
    "django.middleware.security.SecurityMiddleware",
    # Варта active defense (before everything else — block threats early)
    "syn.varta.middleware.VartaMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    # HTTP telemetry (early — captures full middleware chain timing)
    "syn.log.middleware.PerformanceMiddleware",
    # Synara request ID (early — correlation needs this)
    "syn.api.middleware.SynRequestIdMiddleware",
    # CORS (before API surface — ensures 406/error responses include CORS headers)
    "corsheaders.middleware.CorsMiddleware",
    # Synara API surface (API-002 §8-9 — needs syn_request_id from above)
    "syn.api.middleware.APIHeadersMiddleware",
    "syn.api.middleware.IdempotencyMiddleware",
    # Safety subdomain routing (safety.svend.ai → /app/safety/)
    "accounts.middleware.SafetySubdomainMiddleware",
    # Svend + Django standard
    "accounts.middleware.NoCacheDynamicMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    # API key credential resolver (SEC-001 §4.5) — resolves Bearer sv_... → request.user
    "accounts.api_key_auth.APIKeyAuthMiddleware",
    "accounts.middleware.SiteVisitMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    # Synara infrastructure (after auth, before business logic)
    "syn.synara.middleware.csp.ContentSecurityPolicyMiddleware",
    "syn.synara.middleware.tenant.TenantIsolationMiddleware",
    "syn.log.middleware.CorrelationMiddleware",
    "syn.log.middleware.AuditLoggingMiddleware",
    "syn.api.middleware.ErrorEnvelopeMiddleware",
    # Svend business middleware
    "accounts.middleware.SubscriptionMiddleware",
    "accounts.middleware.QueryLimitMiddleware",
]

ROOT_URLCONF = "svend.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "svend.wsgi.application"
ASGI_APPLICATION = "svend.asgi.application"

# Database — PostgreSQL only
DATABASES = {"default": dj_database_url.parse(config.database_url)}
DATABASES["default"]["CONN_MAX_AGE"] = 60  # Reuse connections for 60s (vs 0 = new per request)

# Auth
AUTH_USER_MODEL = "accounts.User"
LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/app/"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# Password hashing — Argon2 primary (memory-hard, SOC 2 CC6.1)
# Requires: pip install django[argon2]
PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.Argon2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher",
]

# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [
    BASE_DIR / "static",
    Path(importlib.import_module("forgeviz").__file__).parent / "static",
    Path(importlib.import_module("forgerack").__file__).parent / "static",
]
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage"},
}

# Media files (user uploads)
MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

# Default primary key
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# CORS
CORS_ALLOWED_ORIGINS = config.get_cors_origins()
CORS_ALLOW_CREDENTIALS = True

# Session security (production settings)
SESSION_COOKIE_SECURE = not DEBUG
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_AGE = 28800  # 8 hours (SOC 2 CC6.1/CC6.6)
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
CSRF_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_HTTPONLY = False
X_FRAME_OPTIONS = "DENY"  # Explicit — XFrameOptionsMiddleware active (SOC 2 CC6.1)

# HTTPS hardening (production only — Caddy handles TLS)
if not DEBUG:
    SECURE_HSTS_SECONDS = 63072000  # 2 years
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_SSL_REDIRECT = True
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# Field-level encryption key (loaded from ~/.svend_encryption_key)
FIELD_ENCRYPTION_KEY = config.field_encryption_key

# REST Framework
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.UserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "user": f"{config.rate_limit_per_minute}/minute",
    },
    # Cloudflare is the single proxy in front of Gunicorn. This tells DRF
    # to use the *rightmost* entry in X-Forwarded-For (the one Cloudflare
    # appended) instead of the leftmost (which the client controls).
    # Without this, all anonymous rate limits are bypassable via a fake
    # X-Forwarded-For header.
    "NUM_PROXIES": 1,
}

# Tempora (distributed task scheduling)
TEMPORA_CLUSTER_SECRET = hashlib.sha256((config.secret_key + ":tempora-cluster").encode()).hexdigest()
TEMPORA_NODE_ID = "svend-1"
TEMPORA_SETTINGS = {
    "election_timeout_min": 150,
    "election_timeout_max": 300,
    "heartbeat_interval": 50,
}

# ---- Synara Infrastructure Settings ----
# Audit trail: which paths/methods to log
AUDIT_PATHS = ["/api/"]
AUDIT_METHODS = ["POST", "PUT", "PATCH", "DELETE"]

# Content Security Policy (allow unsafe-inline — Svend uses inline JS/CSS)
# Values are lists per CSP middleware API (joined with spaces)
CONTENT_SECURITY_POLICY = {
    "default-src": ["'self'"],
    "script-src": [
        "'self'",
        "'unsafe-inline'",
        "'unsafe-eval'",
        "https://js.stripe.com",
        "https://cdn.jsdelivr.net",
        "https://cdnjs.cloudflare.com",
        "https://unpkg.com",
        "https://cdn.plot.ly",
        "https://static.cloudflareinsights.com",
    ],
    "style-src": [
        "'self'",
        "'unsafe-inline'",
        "https://cdn.jsdelivr.net",
        "https://fonts.googleapis.com",
    ],
    "font-src": ["'self'", "https://fonts.gstatic.com", "https://cdn.jsdelivr.net"],
    "img-src": ["'self'", "data:", "blob:", "https:"],
    "connect-src": ["'self'", "https://api.stripe.com", "https://cdn.plot.ly"],
    "frame-src": ["https://js.stripe.com", "https://hooks.stripe.com"],
    "object-src": ["'none'"],
    "base-uri": ["'self'"],
}

# Tenant isolation (disabled — Svend has individual + optional enterprise tenants)
TENANT_ISOLATION_ENABLED = False

# Cognitive scheduler (syn.sched)
SCHEDULER_WORKER_COUNT = 2
SCHEDULER_DEFAULT_TIMEOUT = 300  # seconds
SCHEDULER_MAX_RETRIES = 3
SCHEDULER_DEAD_LETTER_RETENTION_DAYS = 30

# Logging
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{asctime} {levelname} {name} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "svend.log"),
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
            "formatter": "verbose",
        },
        "security": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "security.log"),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 10,
            "formatter": "verbose",
        },
        "varta": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "/var/log/svend/varta.log",
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 10,
            "formatter": "verbose",
        },
    },
    "loggers": {
        "django": {
            "handlers": ["console", "file"],
            "level": "WARNING",
        },
        "django.security": {
            "handlers": ["console", "security"],
            "level": "WARNING",
            "propagate": False,
        },
        "agents_api": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
        "api": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
        "forge": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
        # Synara infrastructure loggers
        "syn": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
        "syn.audit": {
            "handlers": ["console", "security"],
            "level": "INFO",
            "propagate": False,
        },
        # Варта active defense
        "syn.varta": {
            "handlers": ["console", "security"],
            "level": "INFO",
            "propagate": False,
        },
        "syn.varta.actions": {
            "handlers": ["varta"],
            "level": "WARNING",
            "propagate": False,
        },
        "syn.sched": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}

# Kjerne pipeline path
KJERNE_PATH = config.kjerne_path
INFERENCE_DEVICE = config.device

# Add agent modules to path (computed from BASE_DIR which is always correct)
# Note: root core/ is NOT added here — it would shadow Django's core app.
# Agent code handles its own core.* imports via sys.path in agent processes.
# Parent of agents/ so `from agents.experimenter.stats import ...` resolves
_AGENTS_PARENT = str(BASE_DIR.parent)  # services/svend/
if _AGENTS_PARENT not in sys.path:
    sys.path.insert(0, _AGENTS_PARENT)

# Stripe billing
STRIPE_SECRET_KEY = config.stripe_secret_key
STRIPE_WEBHOOK_SECRET = config.stripe_webhook_secret
STRIPE_PRICE_ID_PRO = config.stripe_price_id_pro


# Evidence integration — problem-solving tools → core.Evidence
# Flip to False to disable all tool → evidence hooks (rollback switch).
EVIDENCE_INTEGRATION_ENABLED = True

# Pipeline selection
SVEND_USE_SYNARA = config.use_synara  # True = Synara (alpha-ready), False = MoE
SVEND_USE_OPEN_SOURCE_LM = config.use_open_source_lm
SVEND_OPEN_SOURCE_LM_MODEL = config.open_source_lm_model

# Coder pipeline (Qwen-Coder for computation)
SVEND_ENABLE_CODER = config.enable_coder
SVEND_CODER_MODEL = config.coder_model

# Flywheel / Opus escalation
ANTHROPIC_API_KEY = config.anthropic_api_key
FLYWHEEL_CONFIDENCE_THRESHOLD = config.flywheel_confidence_threshold
FLYWHEEL_AUTO_ESCALATE_THRESHOLD = config.flywheel_auto_escalate_threshold

# Email settings
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = config.email_host
EMAIL_PORT = config.email_port
EMAIL_HOST_USER = config.email_host_user
EMAIL_HOST_PASSWORD = config.email_host_password
EMAIL_USE_TLS = config.email_use_tls
DEFAULT_FROM_EMAIL = config.email_from
SERVER_EMAIL = config.email_from
