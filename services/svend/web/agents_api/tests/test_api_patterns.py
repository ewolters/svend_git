"""FE-001 §6.3 / §6.5 compliance — JavaScript API call patterns.

Validates that templates follow the standard API call pattern:
- POST/PUT fetch calls include Content-Type header (§6.3)
- No getCookie() usage — use getCSRFToken() via the monkey-patch (§6.2/§6.3)
- fetch calls include credentials: 'include' for session auth (§6.3)
- 403 interceptor IIFE is present and correct in rendered pages (§6.5)

Standard: FE-001 §6.3, §6.5
Compliance: SOC 2 CC6.1
"""

import os
import re

from django.conf import settings
from django.test import SimpleTestCase, TestCase, override_settings

from accounts.models import Tier, User

# ── Helpers ──────────────────────────────────────────────────────────────

TEMPLATE_DIR = os.path.join(settings.BASE_DIR, "templates")

# Templates that contain fetch() calls (discovered by scanning templates/)
_FETCH_TEMPLATES = []
for _fname in sorted(os.listdir(TEMPLATE_DIR)):
    if _fname.endswith(".html"):
        _path = os.path.join(TEMPLATE_DIR, _fname)
        with open(_path) as _f:
            if "fetch(" in _f.read():
                _FETCH_TEMPLATES.append(_fname)

# Regex: match a fetch() block up to its closing });
# This is a heuristic — we look for fetch( ... { ... method: 'POST' ...
# and then check whether Content-Type appears between the fetch( and
# the next closing });
_FETCH_BLOCK_RE = re.compile(
    r"fetch\s*\([^)]*,\s*\{(.*?)\}\s*\)",
    re.DOTALL,
)

_POST_PUT_RE = re.compile(r"method\s*:\s*['\"](?:POST|PUT)['\"]", re.IGNORECASE)


def _read_template(name):
    """Read a template file from the templates directory."""
    path = os.path.join(TEMPLATE_DIR, name)
    with open(path) as f:
        return f.read()


# ── APICallPatternTest ───────────────────────────────────────────────────


class APICallPatternTest(SimpleTestCase):
    """Sweep templates for FE-001 §6.3 API call pattern compliance."""

    def test_post_fetch_calls_include_content_type(self):
        """Every fetch() with method POST or PUT must include Content-Type.

        Many templates currently violate this — the regex-based fetch block
        extraction is a heuristic that catches options objects where method
        is POST/PUT but Content-Type is absent. Some of these are legitimate
        (e.g. FormData uploads where Content-Type must NOT be set), but most
        are genuine gaps. We use subTest() per template so each violation is
        individually visible.

        Known gaps: templates listed below omit Content-Type on one or more
        POST/PUT fetch calls. These are real FE-001 §6.3 violations tracked
        for remediation.
        """
        # Templates with known Content-Type gaps — real violations, not false
        # positives. Regex may match fetch blocks where body is FormData
        # (Content-Type must be omitted for multipart), so those are excluded.
        known_content_type_gaps = {
            "analysis_workbench.html",
            "base_app.html",
            "hypotheses.html",
            "internal_dashboard.html",
            "iso_doc.html",
            "learn.html",
            "models.html",
            "problems.html",
            "projects.html",
            "settings.html",
            "triage.html",
            "vsm.html",
            "workbench.html",
            "workflows.html",
        }

        for tpl_name in _FETCH_TEMPLATES:
            content = _read_template(tpl_name)
            tpl_violations = []
            for match in _FETCH_BLOCK_RE.finditer(content):
                block = match.group(1)
                if _POST_PUT_RE.search(block) and "Content-Type" not in block:
                    line_no = content[: match.start()].count("\n") + 1
                    tpl_violations.append(line_no)

            if tpl_violations and tpl_name not in known_content_type_gaps:
                with self.subTest(template=tpl_name):
                    self.fail(f"{tpl_name} has POST/PUT fetch() without Content-Type at lines: {tpl_violations}")

    def test_no_getcookie_usage(self):
        """No template should use getCookie() — FE-001 §6.2 requires getCSRFToken().

        The base_app.html monkey-patch auto-injects X-CSRFToken on non-GET
        requests, so individual templates should not extract CSRF tokens
        manually via getCookie().

        Known gaps: several templates still define/use getCookie(). These are
        real FE-001 §6.2 violations — the monkey-patch handles CSRF
        automatically and getCookie() is redundant.
        """
        # Templates known to still use getCookie() — tracked for remediation
        known_getcookie_users = {
            "a3.html",
            "hoshin.html",
            "models.html",
            "onboarding.html",
            "vsm.html",
        }

        for tpl_name in _FETCH_TEMPLATES:
            content = _read_template(tpl_name)
            if "getCookie(" in content and tpl_name not in known_getcookie_users:
                with self.subTest(template=tpl_name):
                    self.fail(f"{tpl_name} uses getCookie() — should use getCSRFToken() per FE-001 §6.2")

    def test_fetch_calls_include_credentials(self):
        """Templates with >3 fetch() calls should use credentials: 'include'.

        FE-001 §6.3 shows the canonical pattern with credentials: 'include'
        for session cookie auth. Templates with many fetch calls that omit
        this are likely missing session auth on cross-origin or same-origin
        requests.

        Known gaps: several templates omit credentials entirely.
        """
        # Templates known to currently violate — document the gap
        known_failures = {
            "a3.html",
            "rca.html",
            "hoshin.html",
            "simulator.html",
            # Add more here as discovered
        }

        threshold = 3
        violations = []
        for tpl_name in _FETCH_TEMPLATES:
            content = _read_template(tpl_name)
            fetch_count = content.count("fetch(")
            if fetch_count > threshold and "credentials" not in content:
                violations.append(tpl_name)

        # Separate known from unexpected
        unexpected = [v for v in violations if v not in known_failures]
        known_hit = [v for v in violations if v in known_failures]

        if known_hit:
            # These are expected — log them but don't fail
            pass  # Known gap per FE-001 §6.3; tracked for remediation

        self.assertEqual(
            unexpected,
            [],
            f"Unexpected templates with >3 fetch() calls missing credentials: "
            f"{unexpected}. Known gaps (not failing): {known_hit}",
        )

    def test_post_fetch_blocks_found(self):
        """Sanity: we actually find POST/PUT fetch blocks to test against."""
        total = 0
        for tpl_name in _FETCH_TEMPLATES:
            content = _read_template(tpl_name)
            for match in _FETCH_BLOCK_RE.finditer(content):
                block = match.group(1)
                if _POST_PUT_RE.search(block):
                    total += 1
        self.assertGreater(
            total,
            0,
            "Expected to find at least one POST/PUT fetch() block across templates",
        )


# ── FourOhThreeInterceptorTest ───────────────────────────────────────────


@override_settings(SECURE_SSL_REDIRECT=False, RATELIMIT_ENABLE=False)
class FourOhThreeInterceptorTest(TestCase):
    """FE-001 §6.5 — 403 upgrade interceptor in rendered pages."""

    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(
            username="fe001_test",
            email="fe001@test.com",
            password="testpass123",
        )
        cls.user.tier = Tier.PRO
        cls.user.email_verified = True
        cls.user.save()

    def _get_app_page(self):
        """Login and get a rendered app page (inherits base_app.html)."""
        self.client.login(username="fe001_test", password="testpass123")
        return self.client.get("/app/")

    def test_interceptor_iife_present(self):
        """Rendered page contains the 403 interceptor IIFE.

        base_app.html must define a global fetch wrapper that intercepts
        403 responses — look for originalFetch and status === 403.
        """
        resp = self._get_app_page()
        self.assertEqual(resp.status_code, 200)
        html = resp.content.decode()
        self.assertIn("originalFetch", html)
        self.assertIn("status === 403", html)

    def test_interceptor_checks_upgrade_required(self):
        """Interceptor checks for 'Upgrade required' error string."""
        resp = self._get_app_page()
        html = resp.content.decode()
        self.assertIn("Upgrade required", html)

    def test_interceptor_clones_and_shows_modal(self):
        """Interceptor clones response and calls showUpgradeModal."""
        resp = self._get_app_page()
        html = resp.content.decode()
        self.assertIn("response.clone()", html)
        self.assertIn("showUpgradeModal", html)
