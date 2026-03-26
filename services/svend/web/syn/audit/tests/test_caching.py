"""
CACHE-001 compliance tests: Caching Patterns & HTTP Cache Control.

Tests verify HTTP cache headers, static file storage, in-memory cache
bounds, CDN resource integrity, and compliance check registration.

Standard: CACHE-001
Compliance: SOC 2 CC6.1, CC7.2
"""

import importlib
import re
from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase, TestCase

from syn.audit.compliance import (
    _CACHE_MEMORY_BOUNDS,
    ALL_CHECKS,
    _check_cache_cdn_versions,
    _check_cache_idempotency_ttl,
    _check_cache_memory_bounds,
    _check_cache_middleware,
    _check_cache_whitenoise_storage,
)

WEB_ROOT = Path(settings.BASE_DIR)


class CacheMiddlewareTest(SimpleTestCase):
    """CACHE-001 §4: HTTP cache control middleware."""

    def test_no_cache_middleware_present(self):
        """NoCacheDynamicMiddleware is in MIDDLEWARE and sets Cache-Control on responses."""
        from accounts.middleware import NoCacheDynamicMiddleware

        self.assertIn(
            "accounts.middleware.NoCacheDynamicMiddleware",
            settings.MIDDLEWARE,
            "NoCacheDynamicMiddleware not in MIDDLEWARE",
        )
        # Functional: verify the middleware sets Cache-Control on a bare response
        from django.http import HttpRequest, HttpResponse

        def dummy_get_response(request):
            return HttpResponse("ok")

        mw = NoCacheDynamicMiddleware(dummy_get_response)
        response = mw(HttpRequest())
        self.assertIn("no-store", response["Cache-Control"])

    def test_middleware_position(self):
        """WhiteNoise comes before NoCacheDynamic in MIDDLEWARE."""
        issues = _check_cache_middleware(WEB_ROOT)
        self.assertEqual(issues, [], f"Middleware issues: {issues}")


class StaticCacheTest(SimpleTestCase):
    """CACHE-001 §4.2: Static file caching."""

    def test_whitenoise_storage_backend(self):
        """STORAGES uses CompressedManifestStaticFilesStorage."""
        issues = _check_cache_whitenoise_storage()
        self.assertEqual(issues, [], f"Storage issues: {issues}")

    def test_whitenoise_in_middleware(self):
        """WhiteNoise is early in the middleware stack."""
        from syn.audit.compliance import _CACHE_WHITENOISE_MIDDLEWARE

        middleware = settings.MIDDLEWARE
        wn_idx = middleware.index(_CACHE_WHITENOISE_MIDDLEWARE)
        self.assertLess(wn_idx, 4, f"WhiteNoise at position {wn_idx}, should be early")


class AppCacheTest(SimpleTestCase):
    """CACHE-001 §5: Application-level caching."""

    def test_idempotency_ttl_bounded(self):
        """Idempotency cache TTL is bounded (≤48h)."""
        issues = _check_cache_idempotency_ttl()
        self.assertEqual(issues, [], f"Idempotency issues: {issues}")

    def test_idempotency_ttl_value(self):
        """Idempotency TTL is 24 hours."""
        from syn.api.middleware import IDEMPOTENCY_TTL_HOURS

        self.assertEqual(IDEMPOTENCY_TTL_HOURS, 24)


class MemoryCacheTest(SimpleTestCase):
    """CACHE-001 §6: In-memory caches have size bounds."""

    def _assert_cache_bounded(self, module_path, attr_name, min_val, max_val):
        """Verify a cache constant exists and is within bounds."""
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        self.assertGreaterEqual(
            value, min_val, f"{module_path}.{attr_name}={value} < {min_val}"
        )
        self.assertLessEqual(
            value, max_val, f"{module_path}.{attr_name}={value} > {max_val}"
        )

    def test_dsw_cache_has_bounds(self):
        """DSW model cache (MODEL_CACHE_MAX_SIZE) is bounded."""
        self._assert_cache_bounded(
            "agents_api.dsw.common", "MODEL_CACHE_MAX_SIZE", 1, 500
        )

    def test_spc_cache_has_bounds(self):
        """SPC parsed data cache (_CACHE_MAX_SIZE) is bounded."""
        self._assert_cache_bounded("agents_api.spc_views", "_CACHE_MAX_SIZE", 1, 1000)

    def test_synara_cache_has_bounds(self):
        """Synara session cache (_SYNARA_CACHE_MAX) is bounded."""
        self._assert_cache_bounded(
            "agents_api.synara_views", "_SYNARA_CACHE_MAX", 1, 500
        )

    def test_interview_cache_has_bounds(self):
        """Interview session cache (_INTERVIEW_CACHE_MAX) is bounded."""
        self._assert_cache_bounded(
            "agents_api.problem_views", "_INTERVIEW_CACHE_MAX", 1, 500
        )

    def test_all_known_caches_checked(self):
        """All known in-memory caches are in the compliance check inventory."""
        issues = _check_cache_memory_bounds(WEB_ROOT)
        self.assertEqual(issues, [], f"Missing cache bounds: {issues}")

    def test_inventory_complete(self):
        """The memory bounds inventory covers all known caches."""
        expected = {
            "agents_api/dsw/common.py",
            "agents_api/problem_views.py",
            "agents_api/spc_views.py",
            "agents_api/synara_views.py",
        }
        self.assertEqual(set(_CACHE_MEMORY_BOUNDS.keys()), expected)


class SessionCacheTest(TestCase):
    """CACHE-001 §5.3: SessionCache JSON-only serialization."""

    def test_session_cache_json_only(self):
        """SessionCache.set stores as JSON value_type, not pickle."""
        from agents_api.cache import SessionCache
        from agents_api.models import CacheEntry

        success = SessionCache.set("test:json_type", {"key": "value"}, namespace="test")
        self.assertTrue(success, "SessionCache.set should return True")

        entry = CacheEntry.objects.get(key="test:json_type")
        self.assertEqual(entry.value_type, "json")
        # Value is JSON-encoded bytes
        raw = bytes(entry.value)
        import json

        self.assertEqual(json.loads(raw), {"key": "value"})
        entry.delete()

    def test_session_cache_has_namespaces(self):
        """SessionCache supports synara and model namespaces."""
        from agents_api.cache import ModelCache, SynaraCache

        self.assertEqual(SynaraCache.NAMESPACE, "synara")
        self.assertEqual(ModelCache.NAMESPACE, "model")

    def test_session_cache_clear_namespace(self):
        """SessionCache.clear_namespace removes all entries in a namespace."""
        from agents_api.cache import SessionCache

        SessionCache.set("test:ns1", "a", namespace="test_ns")
        SessionCache.set("test:ns2", "b", namespace="test_ns")
        deleted = SessionCache.clear_namespace("test_ns")
        self.assertEqual(deleted, 2)


class LRUCacheTest(SimpleTestCase):
    """CACHE-001 §6.3: @lru_cache with explicit maxsize."""

    def test_lru_caches_have_maxsize(self):
        """Known @lru_cache usages have explicit maxsize parameter."""
        files = [
            WEB_ROOT / "svend_config" / "config.py",
            WEB_ROOT / "syn" / "core" / "config.py",
        ]
        for f in files:
            if f.exists():
                src = f.read_text()
                for m in re.finditer(r"@lru_cache\((.*?)\)", src):
                    with self.subTest(file=f.name, match=m.group()):
                        self.assertIn(
                            "maxsize",
                            m.group(1),
                            f"@lru_cache without maxsize in {f.name}",
                        )

    def test_no_unbounded_lru_cache(self):
        """No @lru_cache(maxsize=None) (unbounded) in core files."""
        for f in WEB_ROOT.glob("**/*.py"):
            if ".venv" in str(f) or "site-packages" in str(f):
                continue
            # Skip test files — they contain patterns as string literals
            if "/tests/" in str(f) or str(f).endswith("tests.py"):
                continue
            try:
                src = f.read_text()
            except Exception:
                continue
            for m in re.finditer(r"@lru_cache\(maxsize\s*=\s*None\)", src):
                self.fail(f"Unbounded @lru_cache in {f.relative_to(WEB_ROOT)}")


class BrowserStorageTest(SimpleTestCase):
    """CACHE-001 §7: Browser storage security."""

    def test_localstorage_no_auth_tokens(self):
        """localStorage.setItem calls do not store auth tokens or passwords."""
        base_app = (WEB_ROOT / "templates" / "base_app.html").read_text()
        setitems = re.findall(r"localStorage\.setItem\(['\"](\w+)['\"]", base_app)
        forbidden = {"token", "jwt", "session_id", "password", "api_key"}
        for key in setitems:
            self.assertNotIn(
                key.lower(), forbidden, f"localStorage stores forbidden key: {key}"
            )

    def test_sessionstorage_no_secrets(self):
        """sessionStorage does not store secrets."""
        for tpl in (WEB_ROOT / "templates").glob("*.html"):
            src = tpl.read_text()
            setitems = re.findall(r"sessionStorage\.setItem\(['\"](\w+)['\"]", src)
            forbidden = {"token", "jwt", "password", "api_key", "secret"}
            for key in setitems:
                self.assertNotIn(
                    key.lower(),
                    forbidden,
                    f"sessionStorage forbidden key '{key}' in {tpl.name}",
                )


class CDNResourceTest(SimpleTestCase):
    """CACHE-001 §8: CDN resources."""

    def test_cdn_scripts_version_pinned(self):
        """CDN resources without version pins are tracked as debt."""
        violations = _check_cache_cdn_versions(WEB_ROOT)
        for v in violations:
            self.assertIn("file", v)
            self.assertIn("url", v)


class CheckRegistrationTest(SimpleTestCase):
    """CACHE-001 §11: Compliance check registration."""

    def test_check_registered(self):
        self.assertIn("caching", ALL_CHECKS)

    def test_check_is_callable(self):
        """Caching compliance check function is callable."""
        fn, _category = ALL_CHECKS["caching"]
        self.assertTrue(callable(fn), "check_caching is not callable")

    def test_check_returns_valid_structure(self):
        """Caching compliance check returns dict with status, details, soc2_controls."""
        fn, _category = ALL_CHECKS["caching"]
        result = fn()
        self.assertIn("status", result)
        self.assertIn("details", result)
        self.assertIn("soc2_controls", result)
        self.assertIn(result["status"], ("pass", "warning", "fail"))
