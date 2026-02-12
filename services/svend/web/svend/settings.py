"""Django settings for Svend."""

import sys
from pathlib import Path

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
    # Third-party
    "rest_framework",
    "corsheaders",
    "tempora",
    # Local apps
    "core",  # Foundation: tenants, projects, hypotheses, knowledge graph
    "accounts",
    "chat",
    "api",
    "forge",
    "files",
    "agents_api.apps.AgentsApiConfig",
    "workbench",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    # Svend custom middleware
    "accounts.middleware.SubscriptionMiddleware",
    "accounts.middleware.QueryLimitMiddleware",
    "accounts.middleware.InviteRequiredMiddleware",
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

# Database
if config.database_url.startswith("sqlite"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
            # Force fresh connections to avoid stale reads with multiple workers
            "CONN_MAX_AGE": 0,
        }
    }
else:
    # PostgreSQL for production
    import dj_database_url
    DATABASES = {
        "default": dj_database_url.parse(config.database_url)
    }

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

# Internationalization
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]
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
CSRF_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_HTTPONLY = True

# REST Framework
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "api.auth.CsrfExemptSessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",  # For API testing
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
}

# Tempora (distributed task scheduling)
TEMPORA_CLUSTER_SECRET = config.secret_key
TEMPORA_NODE_ID = "svend-1"
TEMPORA_SETTINGS = {
    "election_timeout_min": 150,
    "election_timeout_max": 300,
    "heartbeat_interval": 50,
}

# Kjerne pipeline path
KJERNE_PATH = config.kjerne_path
INFERENCE_DEVICE = config.device

# Add kjerne to path for imports
sys.path.insert(0, str(KJERNE_PATH.parent))
sys.path.insert(0, str(KJERNE_PATH))  # For svend_coder and other kjerne modules

# Stripe billing
STRIPE_SECRET_KEY = config.stripe_secret_key
STRIPE_WEBHOOK_SECRET = config.stripe_webhook_secret
STRIPE_PRICE_ID_PRO = config.stripe_price_id_pro

# Alpha access control
REQUIRE_INVITE = config.require_invite

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
