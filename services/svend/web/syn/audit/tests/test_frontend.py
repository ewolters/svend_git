"""
Frontend pattern tests for FE-001 compliance.

Tests verify that templates render correctly, CSS variables are defined,
CSRF protection is active, CDN sources are approved, global JS objects
are present, chart colors use the approved brand palette, and widgets
follow required structural patterns.

These tests are linked from FE-001.md via <!-- test: --> hooks and verified
by the standards compliance runner.

Compliance: FE-001 (Frontend Patterns), SOC 2 CC6.1 (CSRF)
"""

import os
import re

from django.test import SimpleTestCase, override_settings

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _get_app_page(client):
    """Fetch /app/ and return decoded HTML content."""
    res = client.get("/app/")
    return res, res.content.decode()


@SECURE_OFF
class TemplateInheritanceTest(SimpleTestCase):
    """FE-001 §4.1: All authenticated app pages extend base_app.html."""

    def test_app_page_renders_200(self):
        """Main app page returns 200."""
        res = self.client.get("/app/")
        self.assertEqual(res.status_code, 200)

    def test_app_pages_contain_base_markers(self):
        """App pages include markers from base_app.html (CSRF, theme IIFE)."""
        _, content = _get_app_page(self.client)
        self.assertIn("X-CSRFToken", content)
        self.assertIn("svend_theme", content)


@SECURE_OFF
class ThemeSystemTest(SimpleTestCase):
    """FE-001 §5: Theme system with CSS custom properties."""

    def setUp(self):
        _, self.content = _get_app_page(self.client)

    def test_six_themes_defined(self):
        """6 themes defined via data-theme selectors."""
        themes = set(re.findall(r'\[data-theme="(\w+)"\]', self.content))
        expected = {"light", "nordic", "sandstone", "midnight", "contrast"}
        missing = expected - themes
        self.assertFalse(missing, f"Missing themes: {missing}")

    def test_data_theme_attribute(self):
        """Theme applied via data-theme on <html> element."""
        self.assertIn("document.documentElement.setAttribute", self.content)
        self.assertIn("data-theme", self.content)

    def test_css_variable_namespaces(self):
        """CSS variables follow --bg-*, --accent-*, --text-* naming."""
        for var in [
            "--bg-primary",
            "--accent-primary",
            "--text-primary",
            "--success",
            "--warning",
            "--error",
            "--border",
        ]:
            self.assertIn(var, self.content, f"Missing CSS variable: {var}")

    def test_theme_persistence_local_storage(self):
        """Theme read from localStorage on page load."""
        self.assertIn("localStorage.getItem('svend_theme')", self.content)


@SECURE_OFF
class CSRFProtectionTest(SimpleTestCase):
    """FE-001 §6.2: CSRF auto-injection on fetch."""

    def setUp(self):
        _, self.content = _get_app_page(self.client)

    def test_csrf_monkey_patch_present(self):
        """window.fetch is overridden to inject X-CSRFToken."""
        self.assertIn("window.fetch = function", self.content)
        self.assertIn("X-CSRFToken", self.content)

    def test_csrf_cookie_extraction(self):
        """getCSRFToken reads from document.cookie."""
        self.assertIn("getCSRFToken", self.content)
        self.assertIn("csrftoken=", self.content)

    def test_csrf_skips_safe_methods(self):
        """CSRF not injected for GET/HEAD/OPTIONS."""
        self.assertIn("'GET'", self.content)
        self.assertIn("'HEAD'", self.content)
        self.assertIn("'OPTIONS'", self.content)


@SECURE_OFF
class CDNLibraryTest(SimpleTestCase):
    """FE-001 §7: CDN dependencies from approved sources."""

    APPROVED_CDNS = {
        "fonts.googleapis.com",
        "fonts.gstatic.com",
        "cdn.jsdelivr.net",
        "unpkg.com",
        "cdn.plot.ly",
        "static.cloudflareinsights.com",
        "js.stripe.com",
        "cdnjs.cloudflare.com",
    }

    REQUIRED_LIBS = [
        ("katex", "cdn.jsdelivr.net"),
        ("chart.js", "cdn.jsdelivr.net"),
        ("marked", "cdn.jsdelivr.net"),
        ("smiles-drawer", "unpkg.com"),
    ]

    def setUp(self):
        _, self.content = _get_app_page(self.client)

    def test_required_libraries_loaded(self):
        """All required CDN libraries present in base template."""
        for lib_name, cdn in self.REQUIRED_LIBS:
            self.assertIn(cdn, self.content, f"Library {lib_name} not loaded from {cdn}")

    def test_no_unapproved_cdn_sources(self):
        """No script/link tags load from unapproved CDN origins."""
        urls = re.findall(r'(?:src|href)=["\']https?://([^/"\']+)', self.content)
        for domain in urls:
            self.assertIn(domain, self.APPROVED_CDNS, f"Unapproved CDN source: {domain}")


@SECURE_OFF
class GlobalUtilitiesTest(SimpleTestCase):
    """FE-001 §6.4: Global JS utilities."""

    def setUp(self):
        _, self.content = _get_app_page(self.client)

    def test_svend_theme_global(self):
        """SvendTheme object globally available."""
        self.assertIn("window.SvendTheme = SvendTheme;", self.content)

    def test_svend_chart_global(self):
        """SvendChart object globally available."""
        self.assertIn("window.SvendChart = SvendChart;", self.content)

    def test_svend_track_global(self):
        """window.svendTrack function defined."""
        self.assertIn("window.svendTrack", self.content)

    def test_sv_toast_global(self):
        """window.svToast function defined."""
        self.assertIn("window.svToast=function", self.content)


@SECURE_OFF
class NoFrameworkTest(SimpleTestCase):
    """FE-001 §6.1: No SPA framework."""

    def setUp(self):
        _, self.content = _get_app_page(self.client)

    def test_no_react(self):
        """No React framework references."""
        self.assertNotIn("react.min.js", self.content.lower())
        self.assertNotIn("ReactDOM", self.content)

    def test_no_vue(self):
        """No Vue framework references."""
        self.assertNotIn("vue.min.js", self.content.lower())
        self.assertNotIn("Vue.createApp", self.content)

    def test_no_angular(self):
        """No Angular framework references."""
        self.assertNotIn("angular.min.js", self.content.lower())
        self.assertNotIn("ng-app", self.content)


@SECURE_OFF
class FormPatternTest(SimpleTestCase):
    """FE-001 §8.3: Form styling patterns."""

    def test_form_group_styling_defined(self):
        """CSS for .form-group defined in base template."""
        _, content = _get_app_page(self.client)
        self.assertIn(".form-group", content)


# =============================================================================
# Color Compliance (FE-001 §5.1, §10.3)
# =============================================================================

# Approved brand palette from STYLE_GUIDE.md and base_app.html :root.
# These are the ONLY hex colors allowed as CSS variable fallbacks in base_app.html.
# Themes may define different values for the same variables.
BRAND_PALETTE = {
    # Dark theme (root)
    "#0a0f0a",
    "#0d120d",
    "#121a12",
    "#1a261a",
    "#4a9f6e",
    "#4a9faf",
    "#8a7fbf",
    "#e8c547",
    "#e89547",
    "#e8efe8",
    "#9aaa9a",
    "#7a8f7a",
    "#d06060",
    # Light theme
    "#f5f7f5",
    "#eef2ee",
    "#e5ebe5",
    "#dce4dc",
    "#ffffff",
    "#2d7a4a",
    "#2a6070",
    "#4a3f6f",
    "#846a0a",
    "#9a580a",
    "#1a2a1a",
    "#4a5a4a",
    "#5f705f",
    "#a03030",
    # Nordic
    "#f2f5f8",
    "#e8edf2",
    "#dde3ea",
    "#d2d9e2",
    "#2a6b50",
    "#1a5f80",
    "#4a3f70",
    "#7a6505",
    "#9a5808",
    "#1a2030",
    "#485060",
    "#5f6878",
    "#b02828",
    # Sandstone
    "#f7f4f0",
    "#ede8e2",
    "#e3ddd5",
    "#d9d2c8",
    "#2a607a",
    "#5a4070",
    "#7a6208",
    "#a06010",
    "#2a2420",
    "#5a524a",
    "#6f675f",
    "#a82a2a",
    # Midnight
    "#0a0a14",
    "#0d0d1a",
    "#12121f",
    "#1a1a2a",
    "#6a7fff",
    "#4a9fdf",
    "#9a6fff",
    "#ffd54f",
    "#ff9547",
    "#e8e8f8",
    "#9a9aaa",
    "#7e7e95",
    "#ff5a5a",
    # Contrast
    "#000000",
    "#0a0a0a",
    "#141414",
    "#1e1e1e",
    "#50c080",
    "#50b0e0",
    "#b090f0",
    "#f0d050",
    "#f0a050",
    "#c0c0c0",
    "#909090",
    "#f06060",
}

# Chart colors defined in SvendTheme.chartColors (6-color palette).
# These are the CSS variable fallbacks used by the chart system.
CHART_COLORS = ["#4a9f6e", "#4a9faf", "#e8c547", "#e89547", "#8a7fbf", "#d06060"]

# Chemistry element colors are exempt from brand palette enforcement.
# SmilesDrawer uses domain-standard colors for atoms (O=red, N=blue, etc.)
CHEMISTRY_EXEMPT = {"#ef4444", "#3b82f6", "#eab308", "#f97316", "#22c55e", "#a855f7"}


@SECURE_OFF
class ColorComplianceTest(SimpleTestCase):
    """FE-001 §5.1/§10.3: Brand color palette enforcement."""

    def setUp(self):
        _, self.content = _get_app_page(self.client)

    def test_chart_colors_defined(self):
        """SvendTheme.chartColors contains the 6-color brand palette."""
        for color in CHART_COLORS:
            self.assertIn(color, self.content, f"Missing chart color in SvendTheme: {color}")

    def test_chart_colors_use_css_vars(self):
        """SvendTheme.chartColors reads from CSS variables, not hardcoded values."""
        self.assertIn("getCssVar('--accent-primary')", self.content)
        self.assertIn("getCssVar('--accent-blue')", self.content)
        self.assertIn("getCssVar('--accent-gold')", self.content)
        self.assertIn("getCssVar('--accent-orange')", self.content)
        self.assertIn("getCssVar('--accent-purple')", self.content)

    def test_svendchart_uses_theme_colors(self):
        """SvendChart wrapper references SvendTheme for all chart types."""
        self.assertIn("SvendTheme.chartColors", self.content)
        self.assertIn("SvendTheme.chartColorsAlpha", self.content)

    def test_all_themes_define_accent_palette(self):
        """Each theme defines the full accent color set."""
        required_vars = [
            "--accent-primary",
            "--accent-blue",
            "--accent-purple",
            "--accent-gold",
            "--accent-orange",
        ]
        # Check that each data-theme block contains all required variables.
        # Split content by data-theme selectors, verify each block.
        theme_blocks = re.split(r'\[data-theme="(\w+)"\]', self.content)
        # theme_blocks[0] is :root (dark), then pairs of (name, css)
        for i in range(2, len(theme_blocks), 2):
            theme_name = theme_blocks[i - 1]
            block = theme_blocks[i].split("}")[0]  # first {} block
            for var in required_vars:
                self.assertIn(var, block, f"Theme '{theme_name}' missing {var}")

    def test_error_pages_use_brand_colors(self):
        """Error pages (400/403/404/500) use brand palette, not Tailwind."""
        templates_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ),
            "templates",
        )
        tailwind_colors = {"#0f1117", "#e2e8f0", "#f59e0b", "#60a5fa"}
        for page in ["400.html", "403.html", "404.html", "500.html"]:
            path = os.path.join(templates_dir, page)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                html = f.read()
            found = set(re.findall(r"#[0-9a-fA-F]{6}", html))
            violations = found & tailwind_colors
            self.assertFalse(
                violations,
                f"{page} contains non-brand Tailwind colors: {violations}. "
                f"Use brand palette from STYLE_GUIDE.md instead.",
            )

    def test_doe_js_no_hardcoded_tailwind(self):
        """DOE JS files use SvendTheme/CSS vars, not hardcoded Tailwind colors."""
        static_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ),
            "static",
            "js",
        )
        tailwind_colors = {"#60a5fa", "#f97316", "#22c55e", "#6b7280", "#3b82f6", "#e2e8f0"}
        for fname in ["doe-power.js", "doe-analysis.js"]:
            path = os.path.join(static_dir, fname)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                js = f.read()
            found = set(re.findall(r"'(#[0-9a-fA-F]{6})'", js))
            violations = found & tailwind_colors
            self.assertFalse(
                violations,
                f"{fname} contains non-brand colors: {violations}. "
                f"Use SvendTheme.chartColors or CSS var() instead.",
            )


@SECURE_OFF
class WidgetComplianceTest(SimpleTestCase):
    """FE-001 §8: Widget structure and required UI components."""

    def setUp(self):
        _, self.content = _get_app_page(self.client)

    def test_toast_container_present(self):
        """Toast notification container exists in base template."""
        self.assertIn('id="sv-toast-container"', self.content)

    def test_upgrade_overlay_present(self):
        """Upgrade modal overlay exists for 403 interception."""
        self.assertIn('id="upgrade-overlay"', self.content)

    def test_toast_types_supported(self):
        """svToast supports success, error, warning, info types."""
        for toast_type in ["success", "error", "warning", "info"]:
            self.assertIn(toast_type, self.content)

    def test_feedback_button_present(self):
        """Feedback button exists for user feedback submission."""
        self.assertIn("feedback", self.content.lower())

    def test_navigation_present(self):
        """Navigation structure exists in base template."""
        self.assertIn("<nav", self.content.lower())

    def test_theme_selector_mechanism(self):
        """Theme switching mechanism is wired (previewTheme or setTheme)."""
        has_preview = "previewTheme" in self.content
        has_set = "setTheme" in self.content
        has_apply = "applyTheme" in self.content
        self.assertTrue(
            has_preview or has_set or has_apply,
            "No theme switching function found (previewTheme/setTheme/applyTheme)",
        )


# =============================================================================
# Emoji Compliance (FE-001 §9.2)
# =============================================================================

# Unicode emoji ranges to scan for.
# Covers common emoji blocks: Emoticons, Dingbats, Misc Symbols, Transport,
# Supplemental Symbols, Flags, and variation selectors.
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # Emoticons
    "\U0001f300-\U0001f5ff"  # Misc Symbols and Pictographs
    "\U0001f680-\U0001f6ff"  # Transport and Map
    "\U0001f900-\U0001f9ff"  # Supplemental Symbols
    "\U0001fa00-\U0001fa6f"  # Chess, extended-A
    "\U0001fa70-\U0001faff"  # Extended-B
    "\U00002702-\U000027b0"  # Dingbats
    "\U0001f1e0-\U0001f1ff"  # Flags
    "\U00002600-\U000026ff"  # Misc Symbols (includes ⚠ U+26A0)
    "\U00002700-\U000027bf"  # Dingbats (includes ✓ ✗)
    "\U0000fe00-\U0000fe0f"  # Variation selectors
    "\U0000200d"  # Zero-width joiner
    "\U000025cb"  # White circle ○
    "\U000025c9"  # Fisheye ◉
    "\U000025cc"  # Dotted circle ◌
    "]+",
)


class EmojiComplianceTest(SimpleTestCase):
    """FE-001 §9.2: No emoji or Unicode pictographs in templates."""

    def _scan_file(self, path):
        """Return list of (line_num, match) for emoji found in file."""
        hits = []
        try:
            with open(path, errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    found = _EMOJI_RE.findall(line)
                    if found:
                        hits.append((i, found))
        except FileNotFoundError:
            pass
        return hits

    def test_no_emoji_in_dashboard(self):
        """internal_dashboard.html contains no emoji or Unicode pictographs."""
        templates_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ),
            "templates",
        )
        path = os.path.join(templates_dir, "internal_dashboard.html")
        hits = self._scan_file(path)
        self.assertEqual(
            len(hits),
            0,
            f"Emoji found in internal_dashboard.html at lines: "
            f"{', '.join(str(h[0]) for h in hits)}. "
            f"Use _svIcons or inline SVG instead (FE-001 §9.2).",
        )

    def test_no_emoji_in_app_templates(self):
        """Core app templates (base_app, base_guest) contain no emoji."""
        templates_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ),
            "templates",
        )
        for tpl in ["base_app.html", "base_guest.html"]:
            path = os.path.join(templates_dir, tpl)
            hits = self._scan_file(path)
            self.assertEqual(
                len(hits),
                0,
                f"Emoji found in {tpl} at lines: "
                f"{', '.join(str(h[0]) for h in hits)}. "
                f"Use inline SVG instead (FE-001 §9.2).",
            )
