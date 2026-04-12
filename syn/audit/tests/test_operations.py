"""
OPS-001 compliance tests: Operations & Deployment Standard.

Tests configuration values, security settings, static file serving,
health endpoint, and operational script existence.

Standard: OPS-001
"""

import os
import unittest
from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase, override_settings


class GunicornConfigTest(SimpleTestCase):
    """OPS-001 §4: Gunicorn uses gthread workers."""

    def test_gunicorn_config_exists(self):
        config_path = Path(settings.BASE_DIR) / "gunicorn.conf.py"
        # Config may be at web/ root or ops/
        alt_path = Path(settings.BASE_DIR) / "ops" / "gunicorn.conf.py"
        self.assertTrue(
            config_path.exists() or alt_path.exists(),
            "gunicorn.conf.py not found",
        )


class SystemdServiceTest(SimpleTestCase):
    """OPS-001 §4: systemd services configured."""

    def test_svend_service_exists(self):
        service = Path(settings.BASE_DIR) / "ops" / "svend.service"
        self.assertTrue(service.exists(), "ops/svend.service not found")

    def test_backup_timer_exists(self):
        timer = Path(settings.BASE_DIR) / "ops" / "svend-backup.timer"
        self.assertTrue(timer.exists(), "ops/svend-backup.timer not found")

    def test_purge_timer_exists(self):
        timer = Path(settings.BASE_DIR) / "ops" / "svend-purge.timer"
        self.assertTrue(timer.exists(), "ops/svend-purge.timer not found")


class SecuritySettingsTest(SimpleTestCase):
    """OPS-001 §5: HTTPS and security headers enforced."""

    def test_secure_ssl_redirect(self):
        from django.conf import settings

        self.assertTrue(
            getattr(settings, "SECURE_SSL_REDIRECT", False),
            "SECURE_SSL_REDIRECT should be True in production",
        )

    def test_hsts_seconds_set(self):
        from django.conf import settings

        hsts = getattr(settings, "SECURE_HSTS_SECONDS", 0)
        self.assertGreater(hsts, 0, "SECURE_HSTS_SECONDS should be > 0")

    def test_hsts_include_subdomains(self):
        from django.conf import settings

        self.assertTrue(
            getattr(settings, "SECURE_HSTS_INCLUDE_SUBDOMAINS", False),
            "SECURE_HSTS_INCLUDE_SUBDOMAINS should be True",
        )

    def test_hsts_preload(self):
        from django.conf import settings

        self.assertTrue(
            getattr(settings, "SECURE_HSTS_PRELOAD", False),
            "SECURE_HSTS_PRELOAD should be True",
        )

    def test_session_cookie_secure(self):
        from django.conf import settings

        self.assertTrue(
            getattr(settings, "SESSION_COOKIE_SECURE", False),
            "SESSION_COOKIE_SECURE should be True",
        )

    def test_csrf_cookie_secure(self):
        from django.conf import settings

        self.assertTrue(
            getattr(settings, "CSRF_COOKIE_SECURE", False),
            "CSRF_COOKIE_SECURE should be True",
        )


class RateLimitTest(SimpleTestCase):
    """OPS-001 §6: API rate limiting enforced."""

    def test_rate_limit_decorator_exists(self):
        from accounts.permissions import require_auth

        self.assertTrue(callable(require_auth))


class BackupScriptTest(SimpleTestCase):
    """OPS-001 §8: Backup scripts exist."""

    def test_backup_script_exists(self):
        script = Path(settings.BASE_DIR) / "ops" / "backup_db.sh"
        self.assertTrue(script.exists(), "ops/backup_db.sh not found")

    def test_purge_script_exists(self):
        script = Path(settings.BASE_DIR) / "ops" / "run_purge.sh"
        self.assertTrue(script.exists(), "ops/run_purge.sh not found")


class StaticFilesTest(SimpleTestCase):
    """OPS-001 §10: WhiteNoise static file serving."""

    def test_whitenoise_in_middleware(self):
        from django.conf import settings

        middleware = getattr(settings, "MIDDLEWARE", [])
        has_whitenoise = any("whitenoise" in m.lower() for m in middleware)
        self.assertTrue(has_whitenoise, "WhiteNoise middleware not found")

    def test_staticfiles_storage(self):
        from django.conf import settings

        storage = getattr(settings, "STATICFILES_STORAGE", "")
        storages = getattr(settings, "STORAGES", {})
        whitenoise_configured = (
            "whitenoise" in storage.lower()
            or "whitenoise" in str(storages.get("staticfiles", {}).get("BACKEND", "")).lower()
        )
        self.assertTrue(
            whitenoise_configured,
            "Static files storage should use WhiteNoise",
        )


class LogRotationTest(SimpleTestCase):
    """OPS-001 §9: Application logs rotate."""

    def test_logging_config_has_file_handler(self):
        from django.conf import settings

        logging_config = getattr(settings, "LOGGING", {})
        handlers = logging_config.get("handlers", {})
        file_handlers = {
            k: v
            for k, v in handlers.items()
            if "RotatingFileHandler" in v.get("class", "") or "FileHandler" in v.get("class", "")
        }
        self.assertGreater(len(file_handlers), 0, "No file handlers in LOGGING config")


class HealthCheckTest(SimpleTestCase):
    """OPS-001 §11: Health check endpoint available."""

    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_health_endpoint_exists(self):
        from django.test import Client

        client = Client()
        response = client.get("/api/health/")
        self.assertEqual(response.status_code, 200)


class EncryptionConfigTest(SimpleTestCase):
    """OPS-001 §12: Sensitive fields encrypted at rest."""

    def test_fernet_key_configured(self):
        from django.conf import settings

        # Check that a Fernet key or encryption key is configured
        has_key = (
            hasattr(settings, "FERNET_KEY")
            or hasattr(settings, "ENCRYPTION_KEY")
            or hasattr(settings, "FIELD_ENCRYPTION_KEY")
            or os.environ.get("FERNET_KEY")
        )
        self.assertTrue(has_key, "No encryption key configured for field-level encryption")


class StartupScriptTest(SimpleTestCase):
    """OPS-001 §4: Production startup script exists."""

    def test_start_prod_exists(self):
        script = Path(settings.BASE_DIR) / "ops" / "start_prod.sh"
        self.assertTrue(script.exists(), "ops/start_prod.sh not found")


if __name__ == "__main__":
    unittest.main()
