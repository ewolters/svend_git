"""
JavaScript convention tests for JS-001 compliance.

Tests verify calculator architecture, workbench patterns, simulator structure,
DOM manipulation conventions, event handling, async patterns, chart usage,
modal lifecycle, data persistence, and toast notifications.

Tier 1: Global sweep tests enforce anti-patterns across ALL templates.
Tier 2: Surface coverage tests verify patterns in major JS-heavy templates.
Tier 3: Structural enforcement tests verify cross-cutting conventions.

These tests are linked from JS-001.md via <!-- test: --> hooks and verified
by the standards compliance runner.

Compliance: JS-001 (JavaScript Conventions), FE-001 §6 (JavaScript Policy)
"""

import os
import re

from django.test import SimpleTestCase

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "templates")
STATIC_JS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "static", "js")


def _read_template(name):
    """Read a template file and return its content."""
    path = os.path.join(TEMPLATE_DIR, name)
    with open(path) as f:
        return f.read()


def _all_templates():
    """Return sorted list of all .html template files in templates/."""
    return sorted(f for f in os.listdir(TEMPLATE_DIR) if f.endswith(".html"))


def _extract_js(content):
    """Extract inline JavaScript from <script> blocks (excludes <script src=...>)."""
    blocks = []
    for m in re.finditer(r"<script([^>]*)>(.*?)</script>", content, re.DOTALL):
        attrs, body = m.group(1), m.group(2)
        if "src=" not in attrs and body.strip():
            blocks.append(body)
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Naming Convention Tests (JS-001 §4)
# ---------------------------------------------------------------------------


class NamingConventionTest(SimpleTestCase):
    """JS-001 §4: Naming conventions for variables, functions, and DOM IDs."""

    def setUp(self):
        self.calcs = _read_template("calculators.html")
        self.wb = _read_template("workbench_new.html")

    def test_global_variables_camelcase(self):
        """Global let/const variables use camelCase (spot check key variables)."""
        # These are the documented globals from JS-001 §6.2
        for var in [
            "currentWorkbench",
            "currentProjectId",
            "uploadedData",
            "modalCallback",
        ]:
            self.assertIn(
                var,
                self.wb,
                f"Expected camelCase global '{var}' in workbench_new.html",
            )
        for var in ["currentCalcId", "calcMeta"]:
            self.assertIn(
                var,
                self.calcs,
                f"Expected camelCase global '{var}' in calculators.html",
            )

    def test_functions_verb_first(self):
        """Key functions follow verb-first naming (add*, show*, calc*, etc.)."""
        # Check that verb-first function patterns exist
        verb_patterns = [
            (self.calcs, r"function\s+calc\w+\("),
            (self.calcs, r"function\s+showCalc\("),
            (self.wb, r"function\s+openModal\("),
            (self.wb, r"function\s+closeModal\("),
            (self.wb, r"function\s+displayDataTable\("),
            (self.wb, r"function\s+showToast\("),
            (self.wb, r"function\s+importFile\("),
        ]
        for content, pattern in verb_patterns:
            self.assertRegex(
                content,
                pattern,
                f"Expected verb-first function matching {pattern}",
            )

    def test_dom_ids_kebab_case(self):
        """DOM element IDs use kebab-case (spot check key IDs)."""
        # Calculator IDs follow layout-{id} pattern
        self.assertRegex(
            self.calcs,
            r'id="layout-[a-z]',
            "Expected kebab-case layout-* IDs in calculators",
        )
        # Workbench uses kebab-case IDs
        for id_val in ["ribbon-data", "modal-title", "modal-body", "data-table"]:
            self.assertIn(
                f'id="{id_val}"',
                self.wb,
                f"Expected kebab-case ID '{id_val}' in workbench",
            )


# ---------------------------------------------------------------------------
# Calculator Architecture Tests (JS-001 §5)
# ---------------------------------------------------------------------------


class CalculatorArchitectureTest(SimpleTestCase):
    """JS-001 §5: Calculator registry, dispatch, pattern, and data bus."""

    def setUp(self):
        self.content = _read_template("calculators.html")

    def test_calc_meta_exists(self):
        """calcMeta registry object exists."""
        self.assertIn("calcMeta", self.content)
        self.assertRegex(
            self.content,
            r"const\s+calcMeta\s*=\s*\{",
            "calcMeta should be declared as const object",
        )

    def test_calc_meta_has_title_desc(self):
        """calcMeta entries have title and desc fields."""
        self.assertRegex(
            self.content,
            r"calcMeta\s*=\s*\{[^}]*title:",
            "calcMeta entries should have title field",
        )
        self.assertRegex(
            self.content,
            r"calcMeta\s*=\s*\{[^}]*desc:",
            "calcMeta entries should have desc field",
        )

    def test_show_calc_function(self):
        """showCalc function exists and dispatches layout-{id}."""
        self.assertRegex(
            self.content,
            r"function\s+showCalc\(",
            "showCalc function must exist",
        )
        # Should reference layout-${id} or layout- pattern
        self.assertIn(
            "layout-",
            self.content,
            "showCalc should dispatch to layout-{id} elements",
        )

    def test_calc_function_pattern(self):
        """Calculator functions follow input→calc→output pattern (spot check)."""
        # Check that calcTakt or similar follows pattern:
        # parseFloat for input, .toFixed for output, Plotly for chart
        self.assertRegex(
            self.content,
            r"parseFloat\(document\.getElementById\(",
            "Calculators should use parseFloat(getElementById) for input",
        )
        self.assertIn(
            ".toFixed(",
            self.content,
            "Calculators should use .toFixed() for formatted output",
        )
        self.assertIn(
            "Plotly.newPlot",
            self.content,
            "Calculators should use Plotly for chart generation",
        )

    def test_svendops_bus(self):
        """SvendOps data bus exists with publish/pull/get methods."""
        self.assertRegex(
            self.content,
            r"(const|let|var)\s+SvendOps\s*=",
            "SvendOps bus object must exist",
        )
        self.assertIn(
            "publish(",
            self.content,
            "SvendOps must have publish method",
        )
        self.assertIn(
            "pull(",
            self.content,
            "SvendOps must have pull method",
        )


# ---------------------------------------------------------------------------
# Workbench Architecture Tests (JS-001 §6)
# ---------------------------------------------------------------------------


class WorkbenchArchitectureTest(SimpleTestCase):
    """JS-001 §6: Ribbon menu, state variables, API helpers."""

    def setUp(self):
        self.content = _read_template("workbench_new.html")

    def test_ribbon_structure(self):
        """Workbench has ribbon menu with tabs and content panels."""
        self.assertIn(
            "ribbon-tab",
            self.content,
            "Workbench must have ribbon-tab elements",
        )
        self.assertIn(
            "ribbon-content",
            self.content,
            "Workbench must have ribbon-content panels",
        )
        self.assertIn(
            "ribbon-group",
            self.content,
            "Workbench must have ribbon-group structure",
        )
        self.assertIn(
            "ribbon-btn",
            self.content,
            "Workbench must have ribbon-btn buttons",
        )

    def test_state_variables(self):
        """Required global state variables are declared."""
        for var in [
            "currentWorkbench",
            "currentProjectId",
            "uploadedData",
            "modalCallback",
        ]:
            self.assertIn(
                var,
                self.content,
                f"Required state variable '{var}' must be declared",
            )

    def test_api_helpers(self):
        """apiPost and apiCall helper functions exist."""
        self.assertRegex(
            self.content,
            r"(async\s+)?function\s+apiPost\(",
            "apiPost helper must exist",
        )
        self.assertRegex(
            self.content,
            r"(async\s+)?function\s+apiCall\(",
            "apiCall helper must exist",
        )
        # Should include CSRF handling
        self.assertIn(
            "getCSRFToken",
            self.content,
            "API helpers must use getCSRFToken",
        )
        # Should handle 401
        self.assertIn(
            "401",
            self.content,
            "API helpers must handle 401 status",
        )


# ---------------------------------------------------------------------------
# Simulator Architecture Tests (JS-001 §7)
# ---------------------------------------------------------------------------


class SimulatorArchitectureTest(SimpleTestCase):
    """JS-001 §7: SVG canvas, state machine, requestAnimationFrame."""

    def setUp(self):
        self.content = _read_template("simulator.html")

    def test_svg_canvas(self):
        """Simulator uses SVG canvas with pan/zoom."""
        self.assertIn(
            "canvasZoom",
            self.content,
            "Simulator must have canvasZoom variable",
        )
        self.assertIn(
            "canvasPan",
            self.content,
            "Simulator must have canvasPan variables",
        )
        self.assertIn(
            "transform",
            self.content,
            "Simulator must use SVG transform for pan/zoom",
        )
        # Check for SVG namespace usage
        self.assertIn(
            "http://www.w3.org/2000/svg",
            self.content,
            "Simulator must use SVG namespace for element creation",
        )

    def test_state_machine(self):
        """Simulation uses state machine with requestAnimationFrame."""
        self.assertIn(
            "requestAnimationFrame",
            self.content,
            "Simulator must use requestAnimationFrame for animation loop",
        )
        # Check for simulation state tracking (animFrameId controls tick loop)
        self.assertRegex(
            self.content,
            r"animFrameId|simState|simRunning",
            "Simulator must track simulation state",
        )

    def test_transfer_batch_label(self):
        """CR-1: Transfer Batch label replaces ambiguous Batch Size."""
        self.assertIn(
            "Transfer Batch",
            self.content,
            "Machine properties must use 'Transfer Batch' label (not 'Batch Size')",
        )

    def test_accumulation_mode_label(self):
        """CR-1: Accumulation mode label replaces Batch Proc."""
        self.assertIn(
            "Accumulation",
            self.content,
            "Batch processing toggle must use 'Accumulation' label",
        )

    def test_dedicated_product_property(self):
        """CR-2: Dedicated product dropdown exists in machine properties."""
        self.assertIn(
            "dedicated_product",
            self.content,
            "Machine properties must include dedicated_product for single-product machines",
        )

    def test_quality_by_product_section(self):
        """CR-3: Per-product quality overrides exist."""
        self.assertIn(
            "quality_by_product",
            self.content,
            "Machine properties must support per-product quality rates",
        )

    def test_defect_rate_separate_from_scrap(self):
        """CR-3: Defect rate is a separate field from scrap rate."""
        self.assertIn(
            "defect_rate",
            self.content,
            "Machine must have separate defect_rate field",
        )

    def test_routing_rules_section(self):
        """CR-4: Product routing rules on connections."""
        self.assertIn(
            "routing_rules",
            self.content,
            "Connection properties must support routing_rules for divergent flows",
        )

    def test_no_browser_prompt_dialogs(self):
        """CR-5: No browser prompt() dialogs for operator assignment."""
        # prompt( should not appear outside of comments

        # Find all prompt( calls that aren't in comments
        lines = self.content.split("\n")
        violations = []
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            if "prompt(" in line and "showToast" not in line:
                violations.append(f"Line {i}: {stripped[:80]}")
        self.assertEqual(
            violations,
            [],
            "Browser prompt() dialogs must be replaced with config panels:\n" + "\n".join(violations),
        )

    def test_resource_role_config(self):
        """CR-5: Resource role dropdown exists (operator/maintenance/agv/inspector)."""
        for role in ["operator", "maintenance", "agv_driver", "inspector"]:
            self.assertIn(
                role,
                self.content,
                f"Resource role '{role}' must be configurable in employee panel",
            )

    def test_rto_display(self):
        """CR-5: RTO (Required to Operate) display exists."""
        self.assertIn(
            "RTO",
            self.content,
            "Workforce section must display RTO (Required to Operate) budget",
        )

    def test_shift_assignment(self):
        """CR-5: Shift assignment dropdown exists for resources."""
        self.assertIn(
            "Shift 1",
            self.content,
            "Resource config must include shift assignment (Shift 1/2/3)",
        )


# ---------------------------------------------------------------------------
# DOM Pattern Tests (JS-001 §8)
# ---------------------------------------------------------------------------


class DOMPatternTest(SimpleTestCase):
    """JS-001 §8: DOM access patterns, no jQuery."""

    def test_no_jquery(self):
        """No jQuery usage in main templates."""
        for tmpl in ["calculators.html", "workbench_new.html", "simulator.html"]:
            content = _read_template(tmpl)
            # Check for jQuery patterns
            self.assertNotIn(
                "jQuery(",
                content,
                f"jQuery must not be used in {tmpl}",
            )
            # Check for $( pattern but exclude ${} template literals
            jquery_calls = re.findall(r"(?<!\$)\$\(\s*['\"]", content)
            self.assertEqual(
                len(jquery_calls),
                0,
                f"jQuery $() selector must not be used in {tmpl}: found {jquery_calls[:3]}",
            )

    def test_visibility_pattern(self):
        """Visibility uses classList or display property."""
        content = _read_template("workbench_new.html")
        self.assertIn(
            "classList.add",
            content,
            "Should use classList.add for visibility toggling",
        )
        self.assertIn(
            "classList.remove",
            content,
            "Should use classList.remove for visibility toggling",
        )


# ---------------------------------------------------------------------------
# Security Pattern Tests (JS-001 §8.5, §15)
# ---------------------------------------------------------------------------


class SecurityPatternTest(SimpleTestCase):
    """JS-001 §8.5, §15: XSS prevention, no eval, no document.write."""

    def test_no_eval(self):
        """No eval() usage in main templates."""
        for tmpl in ["calculators.html", "workbench_new.html", "simulator.html"]:
            content = _read_template(tmpl)
            # Match eval( but not .evaluate( or evaluation or eval_
            eval_calls = re.findall(r"(?<!\w)eval\s*\(", content)
            self.assertEqual(
                len(eval_calls),
                0,
                f"eval() must not be used in {tmpl}",
            )

    def test_no_document_write(self):
        """No document.write() in templates."""
        for tmpl in ["calculators.html", "workbench_new.html"]:
            content = _read_template(tmpl)
            self.assertNotIn(
                "document.write(",
                content,
                f"document.write() must not be used in {tmpl}",
            )


# ---------------------------------------------------------------------------
# Event Pattern Tests (JS-001 §9)
# ---------------------------------------------------------------------------


class EventPatternTest(SimpleTestCase):
    """JS-001 §9: onclick usage, addEventListener patterns."""

    def test_onclick_usage(self):
        """onclick is used for simple stateless actions."""
        content = _read_template("calculators.html")
        # Should have onclick handlers on buttons
        onclick_count = len(re.findall(r'onclick="', content))
        self.assertGreater(
            onclick_count,
            50,
            f"Expected >50 onclick handlers in calculators.html, found {onclick_count}",
        )

    def test_addeventlistener_usage(self):
        """addEventListener used for complex interactions."""
        content = _read_template("workbench_new.html")
        ael_count = len(re.findall(r"addEventListener\(", content))
        self.assertGreater(
            ael_count,
            5,
            f"Expected >5 addEventListener calls in workbench_new.html, found {ael_count}",
        )


# ---------------------------------------------------------------------------
# Async Pattern Tests (JS-001 §10)
# ---------------------------------------------------------------------------


class AsyncPatternTest(SimpleTestCase):
    """JS-001 §10: async/await with try/catch error handling."""

    def test_async_await_pattern(self):
        """Workbench uses async/await (not .then chains) for API calls."""
        content = _read_template("workbench_new.html")
        async_count = len(re.findall(r"async\s+function", content))
        self.assertGreater(
            async_count,
            5,
            f"Expected >5 async functions in workbench_new.html, found {async_count}",
        )
        # Should have try/catch for error handling
        try_count = len(re.findall(r"\btry\s*\{", content))
        self.assertGreater(
            try_count,
            3,
            f"Expected >3 try blocks in workbench_new.html, found {try_count}",
        )

    def test_credentials_include(self):
        """Fetch calls include credentials for cookie-based auth."""
        content = _read_template("workbench_new.html")
        self.assertIn(
            "credentials",
            content,
            "Fetch calls must include credentials option",
        )


# ---------------------------------------------------------------------------
# Chart Pattern Tests (JS-001 §11)
# ---------------------------------------------------------------------------


class ChartPatternTest(SimpleTestCase):
    """JS-001 §11: Plotly responsive mode and theme-aware colors."""

    def test_plotly_responsive(self):
        """Plotly charts configured with responsive: true."""
        content = _read_template("calculators.html")
        self.assertIn(
            "Plotly.newPlot",
            content,
            "Calculators must use Plotly.newPlot",
        )
        self.assertIn(
            "responsive: true",
            content,
            "Plotly config must include responsive: true",
        )

    def test_plotly_transparent_bg(self):
        """Plotly layouts use transparent background for theme compatibility."""
        content = _read_template("calculators.html")
        self.assertIn(
            "paper_bgcolor: 'transparent'",
            content,
            "Plotly layout should use transparent paper_bgcolor",
        )


# ---------------------------------------------------------------------------
# Modal Pattern Tests (JS-001 §12)
# ---------------------------------------------------------------------------


class ModalPatternTest(SimpleTestCase):
    """JS-001 §12: openModal/closeModal callback pattern."""

    def test_modal_callback_pattern(self):
        """Workbench has openModal/closeModal with callback mechanism."""
        content = _read_template("workbench_new.html")
        self.assertRegex(
            content,
            r"function\s+openModal\(",
            "openModal function must exist",
        )
        self.assertRegex(
            content,
            r"function\s+closeModal\(",
            "closeModal function must exist",
        )
        self.assertIn(
            "modalCallback",
            content,
            "Modal system must use modalCallback pattern",
        )


# ---------------------------------------------------------------------------
# Persistence Pattern Tests (JS-001 §13)
# ---------------------------------------------------------------------------


class PersistencePatternTest(SimpleTestCase):
    """JS-001 §13: Auto-save wrapping, localStorage, file export."""

    def test_autosave_wrapping(self):
        """Auto-save uses function wrapping pattern (not polling)."""
        content = _read_template("calculators.html")
        # Check for the hookAutoSave or equivalent wrapping pattern
        has_hook = "hookAutoSave" in content or "saveAutoState" in content
        self.assertTrue(
            has_hook,
            "Calculator should have auto-save hook pattern (hookAutoSave or saveAutoState)",
        )

    def test_file_export_pattern(self):
        """File export uses Blob + createObjectURL pattern."""
        content = _read_template("workbench_new.html")
        self.assertIn(
            "Blob(",
            content,
            "File export should use Blob constructor",
        )
        self.assertIn(
            "createObjectURL",
            content,
            "File export should use URL.createObjectURL",
        )


# ---------------------------------------------------------------------------
# Toast Pattern Tests (JS-001 §14)
# ---------------------------------------------------------------------------


class ToastPatternTest(SimpleTestCase):
    """JS-001 §14: svToast (global) and showToast (local) patterns."""

    def test_svtoast_global(self):
        """Global svToast function defined in base_app.html."""
        content = _read_template("base_app.html")
        self.assertIn(
            "svToast",
            content,
            "base_app.html must define svToast global",
        )

    def test_showtoast_local(self):
        """Local showToast function defined in workbench template."""
        content = _read_template("workbench_new.html")
        self.assertRegex(
            content,
            r"function\s+showToast\(",
            "workbench_new.html must define local showToast function",
        )


# ===========================================================================
# TIER 1: GLOBAL SWEEP TESTS (JS-001 §15 — Anti-patterns across ALL templates)
# ===========================================================================


class GlobalAntiPatternSweepTest(SimpleTestCase):
    """JS-001 §15: Sweep ALL templates for prohibited patterns.

    Unlike spot-check tests that verify specific templates, these tests
    discover all templates dynamically and enforce anti-patterns globally.
    New templates are automatically included.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.templates = {}
        for name in _all_templates():
            cls.templates[name] = _read_template(name)

    def test_no_jquery_all_templates(self):
        """§15.1: No jQuery usage in any template."""
        violations = []
        for name, content in self.templates.items():
            if "jQuery(" in content:
                violations.append(f"{name}: jQuery() call found")
            jquery_calls = re.findall(r"(?<!\$)\$\(\s*['\"]", content)
            if jquery_calls:
                violations.append(f"{name}: $() selector found ({len(jquery_calls)}x)")
        self.assertEqual(violations, [], "jQuery violations:\n" + "\n".join(violations))

    def test_no_eval_all_templates(self):
        """§15.3: No eval() in any template."""
        violations = []
        for name, content in self.templates.items():
            js = _extract_js(content)
            eval_calls = re.findall(r"(?<!\w)eval\s*\(", js)
            if eval_calls:
                violations.append(f"{name}: eval() found ({len(eval_calls)}x)")
        self.assertEqual(violations, [], "eval() violations:\n" + "\n".join(violations))

    def test_no_document_write_current_page(self):
        """§15.2: No document.write() on current document.

        Writing to a new window (e.g., printWindow.document.write) for
        print-to-PDF is acceptable since it targets a blank popup.
        """
        violations = []
        for name, content in self.templates.items():
            for m in re.finditer(r"<script([^>]*)>(.*?)</script>", content, re.DOTALL):
                attrs, body = m.group(1), m.group(2)
                if "src=" in attrs or not body.strip():
                    continue
                for dm in re.finditer(r"document\.write\s*\(", body):
                    prefix = body[max(0, dm.start() - 50) : dm.start()]
                    # Allow: someVar.document.write( (new window pattern)
                    if not re.search(r"\w+\.\s*$", prefix):
                        violations.append(f"{name}: bare document.write()")
        self.assertEqual(
            violations,
            [],
            "document.write() violations:\n" + "\n".join(violations),
        )

    def test_no_sync_xhr_all_templates(self):
        """§15.5: No synchronous XMLHttpRequest in any template."""
        violations = []
        for name, content in self.templates.items():
            js = _extract_js(content)
            if "XMLHttpRequest" in js:
                violations.append(f"{name}: XMLHttpRequest found (use fetch)")
            sync_opens = re.findall(r"\.open\s*\([^)]*,\s*false\s*\)", js)
            if sync_opens:
                violations.append(f"{name}: synchronous .open() with false flag")
        self.assertEqual(
            violations,
            [],
            "Sync XHR violations:\n" + "\n".join(violations),
        )

    def test_no_unguarded_error_object_coercion(self):
        """§15.8: No .error property used in string context without safeStr().

        ErrorEnvelopeMiddleware returns .error as an object {code, message, ...}.
        Using .error directly in alert(), textContent, innerHTML, or template
        literals produces [object Object]. All must use safeStr().

        Patterns caught:
        - alert(X.error)  or  alert(X.error || 'fallback')
        - .textContent = X.error
        - ${X.error} in template literals (without safeStr wrapper)
        - throw new Error(X.error)  or  throw new Error(X.error || 'fallback')
        - + X.error  (string concatenation)

        Allowed:
        - if (X.error)  (condition check only)
        - console.log/warn/error(X.error)  (debugging)
        - safeStr(X.error, ...)  (already guarded)
        """
        violations = []
        # Patterns where .error is used in string-coercion context
        # without safeStr() wrapping
        coercion_patterns = [
            # alert(data.error) or alert(data.error || 'x')
            (r"alert\([^)]*(?<!safeStr\()(\w+\.error)\b(?!s|_)\s*(\|\||[,)])", "alert({var}.error)"),
            # .textContent = X.error (word boundary: .error not .errors_today)
            (r"\.textContent\s*=\s*(?!.*safeStr).*(\w+\.error)\b(?!s|_)", "textContent = {var}.error"),
            # throw new Error(X.error) or throw new Error(X.error || 'x')
            (r"throw\s+new\s+Error\((?!safeStr)(\w+\.error)\b(?!s|_)", "throw new Error({var}.error)"),
            # showToast(X.error || 'x') without safeStr
            (r"showToast\((?!safeStr)(\w+\.error)\b(?!s|_)\s*\|\|", "showToast({var}.error ||)"),
            # toast(X.error without safeStr
            (r"(?<!\w)toast\((?!safeStr)(\w+\.error)\b(?!s|_)", "toast({var}.error)"),
        ]
        for name, content in self.templates.items():
            js = _extract_js(content)
            for pattern, desc in coercion_patterns:
                for m in re.finditer(pattern, js):
                    # Skip if the line contains safeStr already
                    line_start = js.rfind("\n", 0, m.start()) + 1
                    line_end = js.find("\n", m.end())
                    line = js[line_start : line_end if line_end > 0 else len(js)]
                    if "safeStr" in line:
                        continue
                    # Skip console.log/warn/error
                    if re.search(r"console\.\w+\(", line):
                        continue
                    violations.append(f"{name}: {desc} — {line.strip()[:100]}")
        self.assertEqual(
            violations,
            [],
            "Unguarded .error object coercion (JS-001 §15.8):\n" + "\n".join(violations),
        )

    def test_no_function_constructor_all_templates(self):
        """§15.3: No new Function() constructor in any template."""
        violations = []
        for name, content in self.templates.items():
            js = _extract_js(content)
            fn_constructors = re.findall(r"new\s+Function\s*\(", js)
            if fn_constructors:
                violations.append(f"{name}: new Function() found ({len(fn_constructors)}x)")
        self.assertEqual(
            violations,
            [],
            "Function constructor violations:\n" + "\n".join(violations),
        )


# ===========================================================================
# TIER 2: MAJOR SURFACE COVERAGE (JS-001 §5-§7, §10)
# ===========================================================================


class AnalysisWorkbenchTest(SimpleTestCase):
    """JS-001: Analysis workbench template (5,950 JS lines)."""

    def setUp(self):
        self.content = _read_template("analysis_workbench.html")

    def test_async_with_error_handling(self):
        """Async functions have try/catch error handling."""
        async_count = len(re.findall(r"async\s+function", self.content))
        try_count = len(re.findall(r"\btry\s*\{", self.content))
        self.assertGreater(
            async_count,
            10,
            f"Expected >10 async functions, found {async_count}",
        )
        self.assertGreater(
            try_count,
            10,
            f"Expected >10 try blocks, found {try_count}",
        )

    def test_plotly_charts(self):
        """Uses Plotly for data visualization."""
        self.assertIn("Plotly.newPlot", self.content)

    def test_csrf_handling(self):
        """CSRF token handling for API calls."""
        self.assertIn("getCSRFToken", self.content)

    def test_localstorage_persistence(self):
        """Uses localStorage for client-side persistence."""
        self.assertIn("localStorage", self.content)

    def test_dom_event_initialization(self):
        """Uses addEventListener with DOMContentLoaded for init."""
        self.assertIn("addEventListener", self.content)
        self.assertIn("DOMContentLoaded", self.content)


class InternalDashboardTest(SimpleTestCase):
    """JS-001: Internal staff dashboard (7,130 JS lines)."""

    def setUp(self):
        self.content = _read_template("internal_dashboard.html")

    def test_api_fetch_helper(self):
        """apiFetch wrapper for authenticated API calls."""
        self.assertIn("apiFetch", self.content)

    def test_chart_rendering(self):
        """Uses makeChart helper for Chart.js rendering."""
        self.assertIn("makeChart", self.content)
        chart_count = len(re.findall(r"makeChart\(", self.content))
        self.assertGreater(
            chart_count,
            10,
            f"Expected >10 makeChart calls, found {chart_count}",
        )

    def test_tab_group_system(self):
        """Dashboard has tab group system with loaders."""
        self.assertIn("loaders", self.content)
        self.assertIn("groups", self.content)

    def test_async_error_handling(self):
        """Async functions have try/catch error handling."""
        async_count = len(re.findall(r"async\s+function", self.content))
        try_count = len(re.findall(r"\btry\s*\{", self.content))
        self.assertGreater(
            async_count,
            30,
            f"Expected >30 async functions, found {async_count}",
        )
        self.assertGreater(
            try_count,
            20,
            f"Expected >20 try blocks, found {try_count}",
        )


class HoshinTest(SimpleTestCase):
    """JS-001: Hoshin Kanri enterprise strategic planning (4,156 JS lines)."""

    def setUp(self):
        self.content = _read_template("hoshin.html")

    def test_async_api_pattern(self):
        """Uses async/await for API calls."""
        async_count = len(re.findall(r"async\s+function", self.content))
        self.assertGreater(
            async_count,
            20,
            f"Expected >20 async functions, found {async_count}",
        )

    def test_plotly_visualization(self):
        """Uses Plotly for dashboard charts."""
        self.assertIn("Plotly.react", self.content)

    def test_csrf_handling(self):
        """CSRF token handling via getCookie pattern."""
        self.assertIn("csrftoken", self.content)

    def test_hash_navigation(self):
        """Uses hash-based navigation for SPA-like routing."""
        self.assertIn("hashchange", self.content)


class LearnTest(SimpleTestCase):
    """JS-001: Learning platform with assessments (5,047 JS lines)."""

    def setUp(self):
        self.content = _read_template("learn.html")

    def test_async_api_pattern(self):
        """Uses async/await with try/catch for API calls."""
        async_count = len(re.findall(r"async\s+function", self.content))
        try_count = len(re.findall(r"\btry\s*\{", self.content))
        self.assertGreater(
            async_count,
            5,
            f"Expected >5 async functions, found {async_count}",
        )
        self.assertGreater(
            try_count,
            5,
            f"Expected >5 try blocks, found {try_count}",
        )

    def test_assessment_timer(self):
        """Assessment system uses timer (setInterval)."""
        self.assertIn("setInterval", self.content)

    def test_progress_tracking_with_plotly(self):
        """Progress tracking with Plotly visualization."""
        self.assertIn("progress", self.content)
        self.assertIn("Plotly.newPlot", self.content)


class WhiteboardTest(SimpleTestCase):
    """JS-001: Collaborative whiteboard/knowledge graph (3,365 JS lines)."""

    def setUp(self):
        self.content = _read_template("whiteboard.html")

    def test_async_api_pattern(self):
        """Uses async/await with try/catch for API calls."""
        async_count = len(re.findall(r"async\s+function", self.content))
        try_count = len(re.findall(r"\btry\s*\{", self.content))
        self.assertGreater(
            async_count,
            5,
            f"Expected >5 async functions, found {async_count}",
        )
        self.assertGreater(
            try_count,
            5,
            f"Expected >5 try blocks, found {try_count}",
        )

    def test_svg_canvas(self):
        """Uses SVG for canvas rendering."""
        self.assertIn("<svg", self.content)

    def test_mouse_interaction(self):
        """Canvas supports mouse interaction (drag/pan)."""
        self.assertIn("mousedown", self.content)
        self.assertIn("mousemove", self.content)


class VSMTest(SimpleTestCase):
    """JS-001: Value stream mapping (2,518 JS lines)."""

    def setUp(self):
        self.content = _read_template("vsm.html")

    def test_async_api_pattern(self):
        """Uses async/await with try/catch for API calls."""
        async_count = len(re.findall(r"async\s+function", self.content))
        try_count = len(re.findall(r"\btry\s*\{", self.content))
        self.assertGreater(
            async_count,
            10,
            f"Expected >10 async functions, found {async_count}",
        )
        self.assertGreater(
            try_count,
            10,
            f"Expected >10 try blocks, found {try_count}",
        )

    def test_svg_rendering(self):
        """Uses SVG for process flow rendering."""
        self.assertIn("<svg", self.content)

    def test_plotly_visualization(self):
        """Uses Plotly for analytics charts."""
        self.assertRegex(self.content, r"Plotly\.(newPlot|react)")


class FMEATest(SimpleTestCase):
    """JS-001: FMEA workbench (1,613 JS lines)."""

    def setUp(self):
        self.content = _read_template("fmea.html")

    def test_async_api_pattern(self):
        """Uses async/await for API calls."""
        async_count = len(re.findall(r"async\s+function", self.content))
        self.assertGreater(
            async_count,
            5,
            f"Expected >5 async functions, found {async_count}",
        )

    def test_rpn_calculation(self):
        """RPN (Risk Priority Number) calculation pattern exists."""
        self.assertRegex(
            self.content,
            r"[Rr][Pp][Nn]",
            "FMEA must have RPN calculation",
        )

    def test_plotly_visualization(self):
        """Uses Plotly for risk visualization."""
        self.assertIn("Plotly.newPlot", self.content)

    def test_study_status_labeled(self):
        """QMS-001 §4.1.0a: Study Status dropdown has explicit label."""
        self.assertIn(
            "Study Status:",
            self.content,
            "FMEA toolbar must label the study status dropdown as 'Study Status:'",
        )

    def test_action_status_labeled(self):
        """QMS-001 §4.1.0a: Action Status dropdown has explicit label."""
        self.assertIn(
            "Action Status",
            self.content,
            "FMEA row edit modal must label the action status dropdown as 'Action Status'",
        )

    def test_action_status_filter_exists(self):
        """FMEA has action status filter for row-level filtering."""
        self.assertIn(
            "setActionFilter",
            self.content,
            "FMEA must have row-level action status filtering",
        )

    def test_list_status_filter_exists(self):
        """FMEA list page has study status filter."""
        self.assertIn(
            "setListFilter",
            self.content,
            "FMEA list page must have study status filtering",
        )


class ModelsTest(SimpleTestCase):
    """JS-001: ML model training/prediction (1,909 JS lines)."""

    def setUp(self):
        self.content = _read_template("models.html")

    def test_async_with_error_handling(self):
        """Async functions have try/catch error handling."""
        async_count = len(re.findall(r"async\s+function", self.content))
        try_count = len(re.findall(r"\btry\s*\{", self.content))
        self.assertGreater(
            async_count,
            5,
            f"Expected >5 async functions, found {async_count}",
        )
        self.assertGreater(
            try_count,
            5,
            f"Expected >5 try blocks, found {try_count}",
        )

    def test_plotly_visualization(self):
        """Uses Plotly for model prediction charts."""
        plotly_count = len(re.findall(r"Plotly\.(newPlot|react)", self.content))
        self.assertGreater(
            plotly_count,
            5,
            f"Expected >5 Plotly calls, found {plotly_count}",
        )

    def test_file_upload(self):
        """Supports file upload via FormData."""
        self.assertIn("FormData", self.content)

    def test_csrf_handling(self):
        """CSRF token handling for API calls."""
        self.assertIn("csrftoken", self.content)


class MediumTemplateSweepTest(SimpleTestCase):
    """JS-001: Sanity checks on medium-sized JS templates (280-2,400 lines).

    Verifies baseline patterns: getElementById for DOM access, async/await
    for API calls, fetch for HTTP communication.
    """

    TEMPLATES = [
        "rca.html",
        "a3.html",
        "problems.html",
        "hypotheses.html",
        "iso.html",
        "iso_doc.html",
        "spc.html",
        "report.html",
        "settings.html",
        "projects.html",
        "workflows.html",
        "coder.html",
        "forecast.html",
        "triage.html",
    ]

    def test_all_use_getelementbyid(self):
        """All medium templates use getElementById for DOM access."""
        for name in self.TEMPLATES:
            content = _read_template(name)
            self.assertIn(
                "getElementById",
                content,
                f"{name}: should use getElementById for DOM access",
            )

    def test_all_use_async_await(self):
        """All medium templates use async/await for API calls."""
        for name in self.TEMPLATES:
            content = _read_template(name)
            self.assertRegex(
                content,
                r"async\s+function",
                f"{name}: should use async functions for API calls",
            )

    def test_all_use_fetch(self):
        """All medium templates use fetch for API communication."""
        for name in self.TEMPLATES:
            content = _read_template(name)
            self.assertIn(
                "fetch(",
                content,
                f"{name}: should use fetch() for API calls",
            )


# ===========================================================================
# TIER 3: STRUCTURAL ENFORCEMENT (JS-001 §8, §10, §11)
# ===========================================================================


class PlotlyComplianceSweepTest(SimpleTestCase):
    """JS-001 §11: All Plotly templates must use responsive mode and theme bg."""

    PLOTLY_TEMPLATES = [
        "calculators.html",
        "workbench_new.html",
        "simulator.html",
        "analysis_workbench.html",
        "hoshin.html",
        "learn.html",
        "spc.html",
        "vsm.html",
        "models.html",
        "fmea.html",
    ]

    def test_all_plotly_templates_responsive(self):
        """Every template using Plotly must have responsive: true."""
        violations = []
        for name in self.PLOTLY_TEMPLATES:
            content = _read_template(name)
            if "Plotly." in content and "responsive: true" not in content:
                violations.append(f"{name}: Plotly without responsive: true")
        self.assertEqual(
            violations,
            [],
            "Plotly responsive violations:\n" + "\n".join(violations),
        )

    def test_all_plotly_templates_have_bgcolor(self):
        """Every Plotly template sets paper_bgcolor for theme compat."""
        violations = []
        for name in self.PLOTLY_TEMPLATES:
            content = _read_template(name)
            if "Plotly." in content and "paper_bgcolor" not in content:
                violations.append(f"{name}: Plotly without paper_bgcolor")
        self.assertEqual(
            violations,
            [],
            "Plotly bgcolor violations:\n" + "\n".join(violations),
        )


class AsyncComplianceSweepTest(SimpleTestCase):
    """JS-001 §10: Async pattern enforcement across all JS-heavy templates."""

    def test_async_templates_have_error_handling(self):
        """Templates with >5 async functions must have at least 1 try/catch."""
        violations = []
        for name in _all_templates():
            content = _read_template(name)
            js = _extract_js(content)
            async_count = len(re.findall(r"async\s+function", js))
            if async_count > 5:
                try_count = len(re.findall(r"\btry\s*\{", js))
                if try_count == 0:
                    violations.append(f"{name}: {async_count} async functions, 0 try blocks")
        self.assertEqual(
            violations,
            [],
            "Async error handling gaps:\n" + "\n".join(violations),
        )

    def test_fetch_templates_use_async(self):
        """Templates with >3 fetch() calls should use async/await pattern."""
        violations = []
        for name in _all_templates():
            content = _read_template(name)
            js = _extract_js(content)
            fetch_count = len(re.findall(r"(?<!\w)fetch\s*\(", js))
            if fetch_count > 3:
                async_count = len(re.findall(r"async\s+function", js))
                if async_count == 0:
                    violations.append(f"{name}: {fetch_count} fetch() calls, no async")
        self.assertEqual(
            violations,
            [],
            "Fetch without async:\n" + "\n".join(violations),
        )
