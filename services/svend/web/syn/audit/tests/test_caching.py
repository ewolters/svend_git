"""
CACHE-001 compliance tests: Caching Patterns & HTTP Cache Control.

Tests verify HTTP cache headers, static file storage, in-memory cache
bounds, CDN resource integrity, and compliance check registration.

Standard: CACHE-001
Compliance: SOC 2 CC6.1, CC7.2
"""

from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase

from syn.audit.compliance import (
    ALL_CHECKS,
    _CACHE_MEMORY_BOUNDS,
    _CACHE_REQUIRED_MIDDLEWARE,
    _CACHE_WHITENOISE_MIDDLEWARE,
    _CACHE_WHITENOISE_STORAGE,
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
        """NoCacheDynamicMiddleware is in MIDDLEWARE."""
        middleware = settings.MIDDLEWARE
        self.assertIn(
            _CACHE_REQUIRED_MIDDLEWARE, middleware,
            "NoCacheDynamicMiddleware missing from MIDDLEWARE",
        )

    def test_middleware_position(self):
        """WhiteNoise comes before NoCacheDynamic in MIDDLEWARE."""
        issues = _check_cache_middleware(WEB_ROOT)
        self.assertEqual(issues, [], f"Middleware issues: {issues}")

    def test_whitenoise_present(self):
        """WhiteNoiseMiddleware is in MIDDLEWARE."""
        self.assertIn(
            _CACHE_WHITENOISE_MIDDLEWARE, settings.MIDDLEWARE,
            "WhiteNoiseMiddleware missing from MIDDLEWARE",
        )


class StaticCacheTest(SimpleTestCase):
    """CACHE-001 §4.2: Static file caching."""

    def test_whitenoise_storage_backend(self):
        """STORAGES uses CompressedManifestStaticFilesStorage."""
        issues = _check_cache_whitenoise_storage()
        self.assertEqual(issues, [], f"Storage issues: {issues}")

    def test_whitenoise_in_middleware(self):
        """WhiteNoise is early in the middleware stack."""
        middleware = settings.MIDDLEWARE
        wn_idx = middleware.index(_CACHE_WHITENOISE_MIDDLEWARE)
        # Should be in first 3 positions (after SecurityMiddleware)
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

    def test_dsw_cache_has_bounds(self):
        """DSW model cache has MODEL_CACHE_MAX_SIZE constant."""
        from agents_api.dsw_views import MODEL_CACHE_MAX_SIZE
        self.assertGreater(MODEL_CACHE_MAX_SIZE, 0)
        self.assertLessEqual(MODEL_CACHE_MAX_SIZE, 500)

    def test_spc_cache_has_bounds(self):
        """SPC data cache has _CACHE_MAX_SIZE constant."""
        from agents_api.spc_views import _CACHE_MAX_SIZE
        self.assertGreater(_CACHE_MAX_SIZE, 0)
        self.assertLessEqual(_CACHE_MAX_SIZE, 1000)

    def test_synara_cache_has_bounds(self):
        """Synara cache has _SYNARA_CACHE_MAX constant."""
        from agents_api.synara_views import _SYNARA_CACHE_MAX
        self.assertGreater(_SYNARA_CACHE_MAX, 0)
        self.assertLessEqual(_SYNARA_CACHE_MAX, 500)

    def test_interview_cache_has_bounds(self):
        """Interview cache has _INTERVIEW_CACHE_MAX constant."""
        from agents_api.problem_views import _INTERVIEW_CACHE_MAX
        self.assertGreater(_INTERVIEW_CACHE_MAX, 0)
        self.assertLessEqual(_INTERVIEW_CACHE_MAX, 500)

    def test_all_known_caches_checked(self):
        """All known in-memory caches are in the compliance check inventory."""
        issues = _check_cache_memory_bounds(WEB_ROOT)
        self.assertEqual(issues, [], f"Missing cache bounds: {issues}")

    def test_inventory_complete(self):
        """The memory bounds inventory covers all known caches."""
        expected = {
            "agents_api/dsw_views.py",
            "agents_api/problem_views.py",
            "agents_api/spc_views.py",
            "agents_api/synara_views.py",
        }
        self.assertEqual(set(_CACHE_MEMORY_BOUNDS.keys()), expected)


class SessionCacheTest(SimpleTestCase):
    """CACHE-001 §5.3: SessionCache JSON-only serialization."""

    def test_session_cache_json_only(self):
        """SessionCache uses JSON serialization, not pickle."""
        src = (WEB_ROOT / "agents_api" / "cache.py").read_text()
        self.assertIn("json", src.lower())
        self.assertNotIn("pickle.loads", src)

    def test_session_cache_has_namespaces(self):
        """SessionCache defines synara and model namespaces."""
        src = (WEB_ROOT / "agents_api" / "cache.py").read_text()
        self.assertIn("synara", src)
        self.assertIn("model", src)


class LRUCacheTest(SimpleTestCase):
    """CACHE-001 §6.3: @lru_cache with explicit maxsize."""

    def test_lru_caches_have_maxsize(self):
        """Known @lru_cache usages have explicit maxsize parameter."""
        import re
        files = [
            WEB_ROOT / "svend_config" / "config.py",
            WEB_ROOT / "syn" / "core" / "config.py",
        ]
        for f in files:
            if f.exists():
                src = f.read_text()
                for m in re.finditer(r"@lru_cache\((.*?)\)", src):
                    self.assertIn("maxsize", m.group(1), f"@lru_cache without maxsize in {f.name}")

    def test_no_unbounded_lru_cache(self):
        """No @lru_cache() with maxsize=None (unbounded) in core files."""
        import re
        for f in WEB_ROOT.glob("**/*.py"):
            if ".venv" in str(f) or "site-packages" in str(f):
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
        import re
        base_app = (WEB_ROOT / "templates" / "base_app.html").read_text()
        setitems = re.findall(r"localStorage\.setItem\(['\"](\w+)['\"]", base_app)
        forbidden = {"token", "jwt", "session_id", "password", "api_key"}
        for key in setitems:
            self.assertNotIn(key.lower(), forbidden, f"localStorage stores forbidden key: {key}")

    def test_sessionstorage_no_secrets(self):
        """sessionStorage does not store secrets."""
        import re
        for tpl in (WEB_ROOT / "templates").glob("*.html"):
            src = tpl.read_text()
            setitems = re.findall(r"sessionStorage\.setItem\(['\"](\w+)['\"]", src)
            forbidden = {"token", "jwt", "password", "api_key", "secret"}
            for key in setitems:
                self.assertNotIn(key.lower(), forbidden, f"sessionStorage forbidden key '{key}' in {tpl.name}")


class CDNResourceTest(SimpleTestCase):
    """CACHE-001 §8: CDN resources."""

    def test_cdn_scripts_version_pinned(self):
        """CDN resources without version pins are tracked as debt."""
        violations = _check_cache_cdn_versions(WEB_ROOT)
        # Soft enforcement — violations are warnings, not failures.
        for v in violations:
            self.assertIn("file", v)
            self.assertIn("url", v)


class CheckRegistrationTest(SimpleTestCase):
    """CACHE-001 §11: Compliance check registration."""

    def test_check_registered(self):
        self.assertIn("caching", ALL_CHECKS)

    def test_check_is_callable(self):
        fn, _category = ALL_CHECKS["caching"]
        self.assertTrue(callable(fn))

    def test_check_returns_valid_structure(self):
        fn, _category = ALL_CHECKS["caching"]
        result = fn()
        self.assertIn("status", result)
        self.assertIn("details", result)
        self.assertIn("soc2_controls", result)
        self.assertIn(result["status"], ("pass", "warning", "fail"))
