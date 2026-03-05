"""
SEC-001 compliance tests: Security Architecture Standard.

Tests middleware ordering, transport security settings, CSP headers,
CORS configuration, rate limiting, tenant isolation, error redaction,
and ownership constraints.

Standard: SEC-001
"""

from django.conf import settings
from django.test import SimpleTestCase, TestCase, override_settings


class MiddlewareOrderTest(SimpleTestCase):
    """SEC-001 §1.4: Middleware stack order matches security-critical sequencing."""

    # Order-sensitive middleware pairs: (earlier, later)
    REQUIRED_ORDER = [
        ("SecurityMiddleware", "SessionMiddleware"),
        ("SessionMiddleware", "CsrfViewMiddleware"),
        ("CsrfViewMiddleware", "AuthenticationMiddleware"),
        ("AuthenticationMiddleware", "TenantIsolationMiddleware"),
        ("TenantIsolationMiddleware", "AuditLoggingMiddleware"),
        ("AuditLoggingMiddleware", "ErrorEnvelopeMiddleware"),
        ("ErrorEnvelopeMiddleware", "SubscriptionMiddleware"),
    ]

    def _middleware_index(self, fragment):
        for i, m in enumerate(settings.MIDDLEWARE):
            if fragment in m:
                return i
        return -1

    def test_security_critical_ordering(self):
        for earlier, later in self.REQUIRED_ORDER:
            idx_early = self._middleware_index(earlier)
            idx_late = self._middleware_index(later)
            self.assertGreater(
                idx_early, -1,
                f"{earlier} not found in MIDDLEWARE",
            )
            self.assertGreater(
                idx_late, -1,
                f"{later} not found in MIDDLEWARE",
            )
            self.assertLess(
                idx_early, idx_late,
                f"{earlier} (idx={idx_early}) must come before {later} (idx={idx_late})",
            )

    def test_security_middleware_is_first(self):
        self.assertIn("SecurityMiddleware", settings.MIDDLEWARE[0])

    def test_error_envelope_after_audit_logging(self):
        idx_audit = self._middleware_index("AuditLoggingMiddleware")
        idx_envelope = self._middleware_index("ErrorEnvelopeMiddleware")
        self.assertLess(idx_audit, idx_envelope)


class TenantMiddlewareTest(SimpleTestCase):
    """SEC-001 §6.1: TenantIsolationMiddleware in middleware stack."""

    def test_tenant_middleware_registered(self):
        tenant_mw = [m for m in settings.MIDDLEWARE if "TenantIsolation" in m]
        self.assertEqual(len(tenant_mw), 1)

    def test_tenant_middleware_class_exists(self):
        from syn.synara.middleware.tenant import TenantIsolationMiddleware

        self.assertTrue(callable(TenantIsolationMiddleware))


class OwnershipConstraintTest(SimpleTestCase):
    """SEC-001 §6.3: Models enforce user XOR tenant ownership via CheckConstraint."""

    def test_project_has_xor_constraint(self):
        from core.models.project import Project

        names = [c.name for c in Project._meta.constraints]
        self.assertIn("project_has_single_owner", names)

    def test_knowledge_graph_has_xor_constraint(self):
        from core.models.graph import KnowledgeGraph

        names = [c.name for c in KnowledgeGraph._meta.constraints]
        self.assertIn("graph_has_single_owner", names)


class HSTSConfigTest(SimpleTestCase):
    """SEC-001 §8.1: HSTS configured for 2 years with includeSubDomains and preload."""

    def test_hsts_seconds_two_years(self):
        self.assertEqual(settings.SECURE_HSTS_SECONDS, 63072000)

    def test_hsts_include_subdomains(self):
        self.assertTrue(settings.SECURE_HSTS_INCLUDE_SUBDOMAINS)

    def test_hsts_preload(self):
        self.assertTrue(settings.SECURE_HSTS_PRELOAD)


class SSLConfigTest(SimpleTestCase):
    """SEC-001 §8.2: SSL redirect and content-type nosniff are enabled."""

    def test_ssl_redirect_enabled(self):
        self.assertTrue(settings.SECURE_SSL_REDIRECT)

    def test_content_type_nosniff(self):
        self.assertTrue(settings.SECURE_CONTENT_TYPE_NOSNIFF)


class CSPHeaderTest(TestCase):
    """SEC-001 §8.3: CSP middleware sets restrictive default-src with explicit allowlist."""

    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_csp_header_present_on_response(self):
        resp = self.client.get("/")
        csp = resp.get("Content-Security-Policy", "")
        self.assertTrue(len(csp) > 0, "Content-Security-Policy header missing")

    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_csp_default_src_is_self(self):
        resp = self.client.get("/")
        csp = resp.get("Content-Security-Policy", "")
        self.assertIn("default-src 'self'", csp)

    @override_settings(SECURE_SSL_REDIRECT=False)
    def test_csp_blocks_object_src(self):
        resp = self.client.get("/")
        csp = resp.get("Content-Security-Policy", "")
        self.assertIn("object-src 'none'", csp)


class CORSConfigTest(SimpleTestCase):
    """SEC-001 §8.4: CORS restricts cross-origin access — no wildcard in production."""

    def test_cors_allow_all_not_enabled(self):
        self.assertFalse(
            getattr(settings, "CORS_ALLOW_ALL_ORIGINS", False),
            "CORS_ALLOW_ALL_ORIGINS must not be True in production",
        )

    def test_cors_allowed_origins_is_explicit_list(self):
        origins = getattr(settings, "CORS_ALLOWED_ORIGINS", [])
        self.assertIsInstance(origins, (list, tuple))
        for origin in origins:
            self.assertNotEqual(origin, "*", "Wildcard origin not allowed")


class TenantRateLimitTest(SimpleTestCase):
    """SEC-001 §9.2: TenantRateLimitMiddleware exists for sliding-window hourly limits."""

    def test_rate_limit_middleware_class_exists(self):
        from syn.synara.middleware.rate_limit import TenantRateLimitMiddleware

        self.assertTrue(callable(TenantRateLimitMiddleware))


class DRFThrottleTest(SimpleTestCase):
    """SEC-001 §9.3: Django REST Framework UserRateThrottle is configured."""

    def test_throttle_classes_configured(self):
        rf = getattr(settings, "REST_FRAMEWORK", {})
        classes = rf.get("DEFAULT_THROTTLE_CLASSES", [])
        self.assertTrue(
            any("UserRateThrottle" in c for c in classes),
            "UserRateThrottle not in DEFAULT_THROTTLE_CLASSES",
        )

    def test_throttle_rate_set(self):
        rf = getattr(settings, "REST_FRAMEWORK", {})
        rates = rf.get("DEFAULT_THROTTLE_RATES", {})
        self.assertIn("user", rates)


class ErrorRedactionTest(SimpleTestCase):
    """SEC-001 §10.1: redact_error_message() strips sensitive patterns."""

    def _redact(self, msg):
        from syn.api.middleware import redact_error_message

        return redact_error_message(msg)

    def test_redacts_password(self):
        self.assertNotIn("hunter2", self._redact("password=hunter2"))

    def test_redacts_api_key(self):
        self.assertNotIn("sk_live_abc", self._redact("api_key=sk_live_abc"))

    def test_redacts_token(self):
        self.assertNotIn("eyJhbG", self._redact("token=eyJhbGciOiJIUzI1NiJ9"))

    def test_redacts_email(self):
        self.assertNotIn("user@example.com", self._redact("contact user@example.com for help"))

    def test_redacts_ssn(self):
        self.assertNotIn("123-45-6789", self._redact("SSN is 123-45-6789"))

    def test_redacts_bearer_token(self):
        self.assertNotIn("abc123token", self._redact("Authorization: Bearer abc123token"))

    def test_redacts_env_vars(self):
        result = self._redact("DATABASE_URL=postgres://user:pass@host/db")
        self.assertNotIn("postgres://", result)

    def test_preserves_safe_text(self):
        safe = "Operation completed successfully"
        self.assertEqual(self._redact(safe), safe)


class RedactionAppliedTest(SimpleTestCase):
    """SEC-001 §10.2: ErrorEnvelopeMiddleware applies redaction before response serialization."""

    def test_error_envelope_imports_redaction(self):
        """ErrorEnvelopeMiddleware module contains redact_error_message."""
        import syn.api.middleware as mw

        self.assertTrue(hasattr(mw, "redact_error_message"))
        self.assertTrue(callable(mw.redact_error_message))

    def test_error_envelope_middleware_exists(self):
        from syn.api.middleware import ErrorEnvelopeMiddleware

        self.assertTrue(callable(ErrorEnvelopeMiddleware))

    def test_error_envelope_in_middleware_stack(self):
        envelope_mw = [m for m in settings.MIDDLEWARE if "ErrorEnvelope" in m]
        self.assertEqual(len(envelope_mw), 1)
