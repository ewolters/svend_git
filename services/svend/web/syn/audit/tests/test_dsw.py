"""
DSW-001 compliance tests: Decision Science Workbench Architecture.

Tests verify stateless dispatch, data source resolution, module split,
Synara belief engine invariants, evidence bridge, result persistence,
feature gating, and IDOR prevention patterns.

Standard: DSW-001
"""

import ast
import inspect
import os
import re
import textwrap
from pathlib import Path

from django.test import SimpleTestCase

# Base paths
WEB_ROOT = Path(os.path.dirname(__file__)).parent.parent.parent
DSW_DIR = WEB_ROOT / "agents_api" / "dsw"
SYNARA_DIR = WEB_ROOT / "agents_api" / "synara"
AGENTS_API = WEB_ROOT / "agents_api"


def _read(path):
    """Read a file, return empty string on failure."""
    try:
        return Path(path).read_text(errors="ignore")
    except Exception:
        return ""


def _read_py(relative):
    """Read a Python file relative to WEB_ROOT."""
    return _read(WEB_ROOT / relative)


# ── §4.1: Stateless Dispatch Pattern ─────────────────────────────────────


class StatelessDispatchTest(SimpleTestCase):
    """DSW-001 §4.1: Analysis requests are stateless."""

    def setUp(self):
        self.dispatch_src = _read(DSW_DIR / "dispatch.py")
        self.assertGreater(len(self.dispatch_src), 0, "dispatch.py not found")

    def test_dispatch_has_run_analysis(self):
        """Main entry point run_analysis exists in dispatch.py."""
        self.assertIn("def run_analysis(", self.dispatch_src)

    def test_run_analysis_is_gated(self):
        """run_analysis uses @gated decorator for auth + rate limiting."""
        self.assertRegex(self.dispatch_src, r"@(gated|rate_limited)\s.*?def run_analysis")

    def test_no_session_state_in_dispatch(self):
        """dispatch.py does not use request.session (stateless)."""
        self.assertNotIn("request.session", self.dispatch_src)

    def test_dispatch_reads_body_from_request(self):
        """run_analysis parses JSON body from request."""
        self.assertIn("json.loads(request.body)", self.dispatch_src)

    def test_dispatch_returns_json_response(self):
        """run_analysis returns JsonResponse."""
        self.assertIn("JsonResponse", self.dispatch_src)

    def test_dispatch_handles_invalid_json(self):
        """run_analysis catches JSONDecodeError."""
        self.assertIn("JSONDecodeError", self.dispatch_src)


# ── §4.2: Data Source Resolution ──────────────────────────────────────────


class DataSourceResolutionTest(SimpleTestCase):
    """DSW-001 §4.2: Data source fallback order."""

    def setUp(self):
        self.dispatch_src = _read(DSW_DIR / "dispatch.py")

    def test_inline_data_source(self):
        """Source 0: inline data from body.data dict."""
        self.assertIn('body.get("data")', self.dispatch_src)

    def test_uploaded_file_source(self):
        """Source 1: uploaded file via data_xxx format."""
        self.assertIn('data_id.startswith("data_")', self.dispatch_src)

    def test_temp_directory_fallback(self):
        """Source 1b: temp directory fallback."""
        self.assertIn("svend_analysis", self.dispatch_src)

    def test_triage_source(self):
        """Source 2: TriageResult cleaned dataset."""
        self.assertIn("TriageResult", self.dispatch_src)

    def test_empty_dataframe_for_simulation(self):
        """Source 3: Empty DataFrame allowed for simulation and bayesian."""
        self.assertRegex(
            self.dispatch_src,
            r'analysis_type\s+in\s+\(\s*"simulation".*"bayesian"',
        )

    def test_inline_data_row_limit(self):
        """Inline data capped at 10,000 rows."""
        self.assertIn("10000", self.dispatch_src)

    def test_no_data_returns_400(self):
        """Missing data returns 400 error."""
        self.assertIn("No data loaded", self.dispatch_src)


# ── §4.3: Module Split Architecture ──────────────────────────────────────


class DispatchRoutingTest(SimpleTestCase):
    """DSW-001 §4.3: Analysis routed through dispatch module."""

    def setUp(self):
        self.dispatch_src = _read(DSW_DIR / "dispatch.py")

    def test_routes_stats(self):
        """Dispatch routes 'stats' to dsw/stats.py."""
        self.assertIn("from .stats import run_statistical_analysis", self.dispatch_src)

    def test_routes_ml(self):
        """Dispatch routes 'ml' to dsw/ml.py."""
        self.assertIn("from .ml import run_ml_analysis", self.dispatch_src)

    def test_routes_spc(self):
        """Dispatch routes 'spc' to dsw/spc.py."""
        self.assertIn("from .spc import run_spc_analysis", self.dispatch_src)

    def test_routes_bayesian(self):
        """Dispatch routes 'bayesian' to dsw/bayesian.py."""
        self.assertIn("from .bayesian import run_bayesian_analysis", self.dispatch_src)

    def test_routes_reliability(self):
        """Dispatch routes 'reliability' to dsw/reliability.py."""
        self.assertIn("from .reliability import run_reliability_analysis", self.dispatch_src)

    def test_routes_simulation(self):
        """Dispatch routes 'simulation' to dsw/simulation.py."""
        self.assertIn("from .simulation import run_simulation", self.dispatch_src)

    def test_routes_viz(self):
        """Dispatch routes 'viz' to dsw/viz.py."""
        self.assertIn("from .viz import run_visualization", self.dispatch_src)

    def test_routes_causal(self):
        """Dispatch routes 'causal' to causal_discovery.py."""
        self.assertIn("from ..causal_discovery import run_causal_discovery", self.dispatch_src)

    def test_routes_drift(self):
        """Dispatch routes 'drift' to drift_detection.py."""
        self.assertIn("from ..drift_detection import run_drift_detection", self.dispatch_src)

    def test_routes_anytime(self):
        """Dispatch routes 'anytime' to anytime_valid.py."""
        self.assertIn("from ..anytime_valid import run_anytime_valid", self.dispatch_src)

    def test_routes_bayes_msa(self):
        """Dispatch routes 'bayes_msa' to msa_bayes.py."""
        self.assertIn("from ..msa_bayes import run_bayes_msa", self.dispatch_src)

    def test_routes_quality_econ(self):
        """Dispatch routes 'quality_econ' to quality_economics.py."""
        self.assertIn("from ..quality_economics import run_quality_econ", self.dispatch_src)

    def test_routes_pbs(self):
        """Dispatch routes 'pbs' to pbs_engine.py."""
        self.assertIn("from ..pbs_engine import run_pbs", self.dispatch_src)

    def test_routes_d_type(self):
        """Dispatch routes 'd_type' to dsw/d_type.py."""
        self.assertIn("from .d_type import run_d_type", self.dispatch_src)

    def test_routes_ishap(self):
        """Dispatch routes 'ishap' to interventional_shap.py."""
        self.assertIn("from ..interventional_shap import run_interventional_shap", self.dispatch_src)

    def test_unknown_type_returns_400(self):
        """Unknown analysis_type returns 400."""
        self.assertIn("Unknown analysis type", self.dispatch_src)

    def test_all_submodules_exist(self):
        """All DSW sub-module files exist on disk."""
        expected = ["stats.py", "ml.py", "spc.py", "bayesian.py",
                    "reliability.py", "simulation.py", "viz.py", "d_type.py",
                    "dispatch.py", "common.py"]
        for f in expected:
            self.assertTrue(
                (DSW_DIR / f).exists(),
                f"DSW sub-module missing: {f}",
            )


# ── §5.1: Synara Cache ───────────────────────────────────────────────────


class SynaraCacheTest(SimpleTestCase):
    """DSW-001 §5.1: In-memory cache with project persistence."""

    def setUp(self):
        self.synara_src = _read(AGENTS_API / "synara_views.py")

    def test_cache_max_defined(self):
        """_SYNARA_CACHE_MAX is defined."""
        self.assertIn("_SYNARA_CACHE_MAX", self.synara_src)

    def test_cache_max_is_128(self):
        """Cache bounded at 128 entries."""
        self.assertIn("_SYNARA_CACHE_MAX = 128", self.synara_src)

    def test_cache_eviction_on_full(self):
        """Cache evicts oldest entry when full."""
        self.assertIn("_synara_cache.pop(next(iter(_synara_cache))", self.synara_src)

    def test_get_synara_exists(self):
        """get_synara function exists."""
        self.assertIn("def get_synara(", self.synara_src)

    def test_save_synara_exists(self):
        """save_synara function exists."""
        self.assertIn("def save_synara(", self.synara_src)

    def test_cache_loads_from_project(self):
        """Cache miss loads from core.Project.synara_state."""
        self.assertIn("project.synara_state", self.synara_src)

    def test_save_persists_to_project(self):
        """save_synara writes to project.synara_state."""
        self.assertRegex(self.synara_src, r"project\.synara_state\s*=\s*synara\.to_dict")


# ── §5.2: Project Resolution ─────────────────────────────────────────────


class ProjectResolutionTest(SimpleTestCase):
    """DSW-001 §5.2: _resolve_project maps workbench_id to core.Project."""

    def setUp(self):
        self.synara_src = _read(AGENTS_API / "synara_views.py")

    def test_resolve_project_exists(self):
        """_resolve_project function defined."""
        self.assertIn("def _resolve_project(", self.synara_src)

    def test_tries_project_first(self):
        """Tries core.Project UUID first."""
        self.assertIn("Project.objects.get(id=workbench_id", self.synara_src)

    def test_tries_problem_second(self):
        """Falls back to Problem UUID → core_project FK."""
        self.assertIn("Problem.objects.get(id=workbench_id", self.synara_src)

    def test_auto_creates_core_project(self):
        """Auto-creates core.Project if Problem has no link."""
        self.assertIn("ensure_core_project()", self.synara_src)

    def test_returns_none_on_failure(self):
        """Returns None if neither resolves."""
        self.assertIn("return None", self.synara_src)

    def test_user_filter_applied(self):
        """Applies user filter to prevent IDOR."""
        self.assertIn('"user": user', self.synara_src)


# ── §5.3: Project Context Guard ──────────────────────────────────────────


class ProjectContextGuardTest(SimpleTestCase):
    """DSW-001 §5.3: Hypothesis creation requires valid project."""

    def setUp(self):
        self.synara_src = _read(AGENTS_API / "synara_views.py")

    def test_require_project_exists(self):
        """_require_project helper function defined."""
        self.assertIn("def _require_project(", self.synara_src)

    def test_require_project_returns_400(self):
        """_require_project returns 400 when no project."""
        # Find the function body
        match = re.search(
            r"def _require_project.*?(?=\ndef |\Z)",
            self.synara_src,
            re.DOTALL,
        )
        self.assertIsNotNone(match, "_require_project not found")
        body = match.group()
        self.assertIn("status=400", body)

    def test_add_hypothesis_calls_require_project(self):
        """add_hypothesis calls _require_project before mutation."""
        match = re.search(
            r"def add_hypothesis.*?(?=\ndef |\Z)",
            self.synara_src,
            re.DOTALL,
        )
        self.assertIsNotNone(match, "add_hypothesis not found")
        body = match.group()
        self.assertIn("_require_project(", body)

    def test_require_project_before_create(self):
        """_require_project called BEFORE create_hypothesis."""
        match = re.search(
            r"def add_hypothesis.*?(?=\ndef |\Z)",
            self.synara_src,
            re.DOTALL,
        )
        body = match.group()
        req_pos = body.find("_require_project(")
        create_pos = body.find("create_hypothesis(")
        self.assertGreater(create_pos, req_pos,
                           "_require_project must be called before create_hypothesis")

    def test_error_message_clear(self):
        """Error message tells user to create/select a study."""
        self.assertIn("No study loaded", self.synara_src)

    def test_all_mutating_endpoints_guarded(self):
        """All hypothesis/evidence/link creation endpoints call _require_project."""
        mutating = ["add_hypothesis", "add_evidence", "add_link"]
        for fn_name in mutating:
            match = re.search(
                rf"def {fn_name}\(.*?(?=\ndef |\Z)",
                self.synara_src,
                re.DOTALL,
            )
            if match:
                self.assertIn(
                    "_require_project(",
                    match.group(),
                    f"{fn_name} does not call _require_project",
                )


# ── §5.4: Save Failure Semantics ─────────────────────────────────────────


class SaveFailureSemanticsTest(SimpleTestCase):
    """DSW-001 §5.4: save_synara failure must propagate as error."""

    def setUp(self):
        self.synara_src = _read(AGENTS_API / "synara_views.py")

    def test_save_returns_bool(self):
        """save_synara returns bool (True/False)."""
        match = re.search(
            r"def save_synara.*?(?=\ndef |\Z)",
            self.synara_src,
            re.DOTALL,
        )
        body = match.group()
        self.assertIn("return True", body)
        self.assertIn("return False", body)

    def test_add_hypothesis_checks_save(self):
        """add_hypothesis checks save_synara return value."""
        match = re.search(
            r"def add_hypothesis.*?(?=\ndef |\Z)",
            self.synara_src,
            re.DOTALL,
        )
        body = match.group()
        self.assertRegex(body, r"if not save_synara\(")

    def test_save_failure_returns_500(self):
        """Save failure returns 500 status."""
        match = re.search(
            r"def add_hypothesis.*?(?=\ndef |\Z)",
            self.synara_src,
            re.DOTALL,
        )
        body = match.group()
        self.assertIn("status=500", body)

    def test_save_failure_message(self):
        """Save failure includes meaningful error message."""
        self.assertIn("Failed to persist", self.synara_src)

    def test_delete_hypothesis_checks_save(self):
        """delete_hypothesis checks save_synara return value."""
        match = re.search(
            r"def delete_hypothesis.*?(?=\ndef |\Z)",
            self.synara_src,
            re.DOTALL,
        )
        if match:
            body = match.group()
            self.assertIn("save_synara(", body)


# ── §6.1: Evidence Bridge ────────────────────────────────────────────────


class EvidenceBridgeTest(SimpleTestCase):
    """DSW-001 §6.1: Analysis results link to Problem as evidence."""

    def setUp(self):
        self.dispatch_src = _read(DSW_DIR / "dispatch.py")

    def test_problem_id_optional(self):
        """problem_id read from request body (optional)."""
        self.assertIn('body.get("problem_id")', self.dispatch_src)

    def test_calls_add_finding(self):
        """Dispatch calls add_finding_to_problem when problem_id provided."""
        self.assertIn("add_finding_to_problem", self.dispatch_src)

    def test_evidence_linking_in_dispatch(self):
        """Evidence linking happens inside dispatch.py, not sub-modules."""
        # Sub-modules should NOT contain add_finding_to_problem
        for module in ["stats.py", "ml.py", "spc.py", "bayesian.py"]:
            src = _read(DSW_DIR / module)
            self.assertNotIn(
                "add_finding_to_problem",
                src,
                f"{module} should not call add_finding_to_problem directly",
            )


# ── §6.2: DSWResult Persistence ──────────────────────────────────────────


class DSWResultPersistenceTest(SimpleTestCase):
    """DSW-001 §6.2: DSWResult stores analysis output."""

    def setUp(self):
        self.models_src = _read(AGENTS_API / "models.py")
        self.dispatch_src = _read(DSW_DIR / "dispatch.py")

    def test_dswresult_model_exists(self):
        """DSWResult class defined in models.py."""
        self.assertIn("class DSWResult(", self.models_src)

    def test_dswresult_has_user_fk(self):
        """DSWResult has user foreign key."""
        # Extract DSWResult class body
        match = re.search(
            r"class DSWResult.*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        body = match.group()
        self.assertIn("user", body)

    def test_dswresult_has_result_data(self):
        """DSWResult stores result content."""
        match = re.search(
            r"class DSWResult.*?(?=\nclass |\Z)",
            self.models_src,
            re.DOTALL,
        )
        body = match.group()
        self.assertTrue(
            "result_data" in body or "content" in body or "EncryptedTextField" in body,
            "DSWResult must store analysis output",
        )

    def test_dispatch_creates_dswresult(self):
        """dispatch.py creates DSWResult when save_result is True."""
        self.assertIn("DSWResult", self.dispatch_src)

    def test_project_id_optional(self):
        """project_id read from body for result linking."""
        self.assertIn('body.get("project_id")', self.dispatch_src)


# ── §7.1: Feature Gating ────────────────────────────────────────────────


class FeatureGatingTest(SimpleTestCase):
    """DSW-001 §7.1: DSW endpoints use @gated/@gated_paid decorators."""

    def setUp(self):
        self.dispatch_src = _read(DSW_DIR / "dispatch.py")
        self.synara_src = _read(AGENTS_API / "synara_views.py")
        self.permissions_src = _read(WEB_ROOT / "accounts" / "permissions.py")

    def test_dispatch_uses_gated(self):
        """run_analysis uses @gated (basic tier access)."""
        self.assertRegex(self.dispatch_src, r"@(gated|rate_limited)\s")

    def test_synara_uses_gated_paid(self):
        """Synara endpoints use @gated_paid (premium tier access)."""
        self.assertIn("@gated_paid", self.synara_src)

    def test_gated_decorator_defined(self):
        """gated decorator exists in permissions.py."""
        self.assertTrue(
            "gated" in self.permissions_src,
            "gated decorator not found in permissions.py",
        )

    def test_gated_paid_decorator_defined(self):
        """gated_paid function defined in permissions.py."""
        self.assertIn("def gated_paid(", self.permissions_src)

    def test_synara_all_endpoints_gated(self):
        """All Synara view functions have @gated_paid."""
        # Find all view functions (def foo(request, ...):)
        view_fns = re.findall(
            r"def (\w+)\(request",
            self.synara_src,
        )
        for fn_name in view_fns:
            if fn_name.startswith("_"):
                continue  # Skip private helpers
            # Check that @gated_paid appears before the function definition
            pattern = rf"@gated_paid\s.*?def {fn_name}\("
            self.assertRegex(
                self.synara_src,
                pattern,
                f"Synara view {fn_name} missing @gated_paid",
            )


# ── §7.2: IDOR Prevention ───────────────────────────────────────────────


class IDORPreventionTest(SimpleTestCase):
    """DSW-001 §7.2: Project lookups filter by request.user."""

    def setUp(self):
        self.synara_src = _read(AGENTS_API / "synara_views.py")

    def test_resolve_project_accepts_user(self):
        """_resolve_project accepts user parameter."""
        self.assertIn("def _resolve_project(workbench_id", self.synara_src)
        match = re.search(r"def _resolve_project\([^)]+\)", self.synara_src)
        self.assertIn("user", match.group())

    def test_add_hypothesis_passes_user(self):
        """add_hypothesis passes request.user to _require_project."""
        match = re.search(
            r"def add_hypothesis.*?(?=\ndef |\Z)",
            self.synara_src,
            re.DOTALL,
        )
        body = match.group()
        self.assertIn("user=request.user", body)

    def test_get_synara_passes_user(self):
        """get_synara calls pass user for IDOR protection."""
        match = re.search(
            r"def add_hypothesis.*?(?=\ndef |\Z)",
            self.synara_src,
            re.DOTALL,
        )
        body = match.group()
        self.assertIn("get_synara(workbench_id, user=request.user)", body)

    def test_save_synara_passes_user(self):
        """save_synara calls pass user for IDOR protection."""
        match = re.search(
            r"def add_hypothesis.*?(?=\ndef |\Z)",
            self.synara_src,
            re.DOTALL,
        )
        body = match.group()
        self.assertIn("save_synara(workbench_id, synara, user=request.user)", body)

    def test_dispatch_filters_triage_by_user(self):
        """Triage data source filters by request.user."""
        dispatch_src = _read(DSW_DIR / "dispatch.py")
        self.assertIn("user=request.user", dispatch_src)


# ── §8: Anti-Patterns ───────────────────────────────────────────────────


class AntiPatternTest(SimpleTestCase):
    """DSW-001 §8: Anti-pattern detection."""

    def setUp(self):
        self.synara_src = _read(AGENTS_API / "synara_views.py")

    def test_no_silent_save_failure(self):
        """No endpoint returns success after save_synara without checking return."""
        # Find all save_synara calls and verify they're in if-not-save patterns
        lines = self.synara_src.split("\n")
        for i, line in enumerate(lines):
            if "save_synara(" in line and "def save_synara" not in line:
                # This line calls save_synara — check it's in an if-check
                context = "\n".join(lines[max(0, i - 1): i + 3])
                has_check = (
                    "if not save_synara" in context
                    or "if save_synara" in context  # positive check
                    or "saved = " in context  # assigned to variable
                )
                # Allow within save_synara's own body
                if "def save_synara" not in "\n".join(lines[max(0, i - 5): i]):
                    self.assertTrue(
                        has_check,
                        f"Line {i + 1}: save_synara() called without checking return value",
                    )


# ── Module Structure ─────────────────────────────────────────────────────


class DSWModuleStructureTest(SimpleTestCase):
    """DSW-001 §1.3: Module structure verification."""

    def test_dsw_package_exists(self):
        """agents_api/dsw/ package exists."""
        self.assertTrue(DSW_DIR.exists())
        self.assertTrue((DSW_DIR / "__init__.py").exists())

    def test_synara_package_exists(self):
        """agents_api/synara/ package exists."""
        self.assertTrue(SYNARA_DIR.exists())

    def test_dispatch_module_exists(self):
        """dispatch.py is the routing entry point."""
        self.assertTrue((DSW_DIR / "dispatch.py").exists())

    def test_common_module_exists(self):
        """common.py provides shared utilities."""
        self.assertTrue((DSW_DIR / "common.py").exists())

    def test_synara_views_exists(self):
        """synara_views.py provides belief engine endpoints."""
        self.assertTrue((AGENTS_API / "synara_views.py").exists())

    def test_dsw_views_exists(self):
        """dsw_views.py provides monolith endpoints."""
        self.assertTrue((AGENTS_API / "dsw_views.py").exists())

    def test_spc_engine_exists(self):
        """SPC engine files exist."""
        self.assertTrue((AGENTS_API / "spc.py").exists())
        self.assertTrue((AGENTS_API / "spc_views.py").exists())

    def test_experimenter_exists(self):
        """DOE experimenter views exist."""
        self.assertTrue((AGENTS_API / "experimenter_views.py").exists())

    def test_permissions_module_exists(self):
        """accounts/permissions.py provides decorators."""
        self.assertTrue((WEB_ROOT / "accounts" / "permissions.py").exists())

    def test_dsw_result_in_models(self):
        """DSWResult defined in agents_api/models.py."""
        models_src = _read(AGENTS_API / "models.py")
        self.assertIn("class DSWResult(", models_src)


# ── Endpoint Decorator Compliance ────────────────────────────────────────


class EndpointDecoratorComplianceTest(SimpleTestCase):
    """DSW-001 §7: All DSW endpoints have auth decorators."""

    def test_dsw_endpoints_data_gated(self):
        """endpoints_data.py uses auth decorators."""
        src = _read(DSW_DIR / "endpoints_data.py")
        if src:
            # Every POST endpoint should have a decorator
            self.assertTrue(
                "@gated" in src or "@rate_limited" in src or "@require_auth" in src or "@gated_paid" in src,
                "endpoints_data.py has no auth decorators",
            )

    def test_dsw_endpoints_ml_gated(self):
        """endpoints_ml.py uses auth decorators."""
        src = _read(DSW_DIR / "endpoints_ml.py")
        if src:
            self.assertTrue(
                "@gated" in src or "@rate_limited" in src or "@require_auth" in src or "@gated_paid" in src,
                "endpoints_ml.py has no auth decorators",
            )

    def test_spc_views_gated(self):
        """spc_views.py uses auth decorators."""
        src = _read(AGENTS_API / "spc_views.py")
        self.assertTrue(
            "@gated" in src or "@rate_limited" in src or "@gated_paid" in src,
            "spc_views.py has no auth decorators",
        )

    def test_experimenter_views_gated(self):
        """experimenter_views.py uses auth decorators."""
        src = _read(AGENTS_API / "experimenter_views.py")
        self.assertTrue(
            "@gated" in src or "@rate_limited" in src or "@gated_paid" in src,
            "experimenter_views.py has no auth decorators",
        )


# =============================================================================
# §8 — Output Standardization — Hardened Tests
# =============================================================================

def _run_analysis(analysis_type, analysis_id, df=None, config=None):
    """Run an analysis through the full pipeline and return standardized result."""
    import pandas as pd
    import numpy as np
    if df is None:
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.normal(50, 2, 100),
            'y': np.random.normal(52, 2, 100),
            'group': (['A'] * 50) + (['B'] * 50),
        })
    if config is None:
        config = {}

    # Import the right module
    if analysis_type == 'stats':
        from agents_api.dsw.stats import run_statistical_analysis
        result = run_statistical_analysis(df, analysis_id, config)
    elif analysis_type == 'spc':
        from agents_api.dsw.spc import run_spc_analysis
        result = run_spc_analysis(df, analysis_id, config)
    elif analysis_type == 'ml':
        from agents_api.dsw.ml import run_ml_analysis
        result = run_ml_analysis(df, analysis_id, config, user=None)
    elif analysis_type == 'viz':
        from agents_api.dsw.viz import run_visualization
        result = run_visualization(df, analysis_id, config)
    else:
        result = {"summary": "test"}

    from agents_api.dsw.standardize import standardize_output
    return standardize_output(result, analysis_type, analysis_id)


class OutputSchemaTest(SimpleTestCase):
    """DSW-001 §8.1: Canonical output schema enforcement."""

    def test_mandatory_keys_present(self):
        """standardize_output() fills all mandatory keys."""
        from agents_api.dsw.standardize import standardize_output, REQUIRED_FIELDS
        result = standardize_output({"summary": "test"}, "stats", "ttest")
        for key in REQUIRED_FIELDS:
            self.assertIn(key, result, f"Missing mandatory key: {key}")

    def test_education_always_present(self):
        """Education key present and non-None after standardization."""
        from agents_api.dsw.standardize import standardize_output
        result = standardize_output({"summary": "test output"}, "stats", "ttest")
        self.assertIsNotNone(result.get("education"),
                             "education is None after standardize — should be filled from centralized store")

    def test_narrative_always_present(self):
        """Narrative generated from summary with non-empty verdict."""
        from agents_api.dsw.standardize import standardize_output
        result = standardize_output({"summary": "Test result line 1\nLine 2"}, "stats", "ttest")
        self.assertIsNotNone(result.get("narrative"))
        self.assertEqual(result["narrative"]["verdict"], "Test result line 1")
        self.assertTrue(len(result["narrative"]["verdict"]) >= 10,
                        "Narrative verdict too short (<10 chars)")

    def test_integration_capability_analysis(self):
        """Cpk analysis produces education, narrative, charts through full pipeline."""
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        df = pd.DataFrame({'measurement': np.random.normal(50, 2, 100)})
        result = _run_analysis('spc', 'capability', df,
                               {'column': 'measurement', 'lsl': 44, 'usl': 56})

        self.assertIsNotNone(result.get("education"),
                             "Cpk analysis missing education after standardization")
        self.assertIn("title", result["education"])
        self.assertTrue(len(result["education"]["content"]) >= 200,
                        f"Cpk education content too shallow: {len(result['education']['content'])} chars")

        self.assertIsNotNone(result.get("narrative"),
                             "Cpk analysis missing narrative")
        self.assertTrue(len(result["narrative"].get("verdict", "")) >= 10,
                        "Cpk narrative verdict too short")

        self.assertTrue(len(result.get("plots", [])) >= 1,
                        "Cpk analysis should produce at least 1 chart")
        for plot in result["plots"]:
            if isinstance(plot, dict) and "layout" in plot:
                self.assertIn("height", plot["layout"],
                              "Cpk chart missing height after chart_defaults")

    def test_integration_ttest_analysis(self):
        """t-test produces education, narrative, evidence grade, shadow through full pipeline."""
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        df = pd.DataFrame({
            'before': np.random.normal(50, 5, 30),
            'after': np.random.normal(55, 5, 30),
        })
        result = _run_analysis('stats', 'ttest', df,
                               {'var1': 'before', 'mu': 50, 'alpha': 0.05})

        self.assertIsNotNone(result.get("education"),
                             "t-test missing education")
        self.assertIsNotNone(result.get("narrative"),
                             "t-test missing narrative")

        # t-test has p_value — should have evidence grade and shadow
        if result.get("p_value") is not None or (result.get("statistics") or {}).get("p_value") is not None:
            self.assertIsNotNone(result.get("evidence_grade"),
                                 "t-test with p-value missing evidence_grade")
            self.assertIsNotNone(result.get("bayesian_shadow"),
                                 "t-test with p-value missing bayesian_shadow")

    def test_integration_regression_analysis(self):
        """Regression produces education, narrative, what-if through full pipeline."""
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        df = pd.DataFrame({'x': x, 'y': 2 * x + np.random.normal(0, 0.5, 50)})
        result = _run_analysis('stats', 'regression', df,
                               {'predictors': ['x'], 'response': 'y'})

        self.assertIsNotNone(result.get("education"),
                             "Regression missing education")
        self.assertIsNotNone(result.get("narrative"),
                             "Regression missing narrative")


class RegistryTest(SimpleTestCase):
    """DSW-001 §8.2: Analysis registry completeness."""

    def test_all_dispatch_types_registered(self):
        """All dispatch route types have at least one registry entry."""
        from agents_api.dsw.registry import ANALYSIS_REGISTRY
        dispatch_src = _read(DSW_DIR / "dispatch.py")
        route_types = set(re.findall(r'analysis_type\s*==\s*"(\w+)"', dispatch_src))
        registry_types = {t for t, _ in ANALYSIS_REGISTRY.keys()}
        for rt in route_types:
            self.assertIn(rt, registry_types,
                          f"Dispatch route type '{rt}' has no registry entries")

    def test_registry_has_required_fields(self):
        """Every registry entry has the required metadata fields."""
        from agents_api.dsw.registry import ANALYSIS_REGISTRY
        required = {"module", "category", "has_pvalue", "effect_type", "shadow_type"}
        for key, entry in ANALYSIS_REGISTRY.items():
            for field in required:
                self.assertIn(field, entry,
                              f"Registry entry {key} missing field '{field}'")


class PostProcessorTest(SimpleTestCase):
    """DSW-001 §8.3: Post-processing pipeline."""

    def test_standardize_called_on_dispatch(self):
        """dispatch.py calls standardize_output before returning."""
        src = _read(DSW_DIR / "dispatch.py")
        self.assertIn("standardize_output", src)
        self.assertIn("from .standardize import standardize_output", src)

    def test_missing_education_filled(self):
        """Post-processor injects education from centralized store."""
        from agents_api.dsw.standardize import standardize_output
        result = standardize_output({}, "stats", "ttest")
        from agents_api.dsw.education import get_education
        edu = get_education("stats", "ttest")
        if edu:
            self.assertIsNotNone(result.get("education"),
                                 "Education exists in centralized store but post-processor did not inject it")

    def test_missing_narrative_generated(self):
        """Post-processor generates narrative from summary."""
        from agents_api.dsw.standardize import standardize_output
        result = standardize_output({"summary": "A detailed test result"}, "stats", "ttest")
        self.assertIsNotNone(result.get("narrative"))

    def test_narrative_has_nonempty_verdict(self):
        """Generated narrative has verdict ≥10 characters."""
        from agents_api.dsw.standardize import standardize_output
        result = standardize_output(
            {"summary": "One-Sample t-Test: t(29) = 2.45, p = 0.021"},
            "stats", "ttest",
        )
        narrative = result.get("narrative")
        self.assertIsNotNone(narrative)
        verdict = narrative.get("verdict", "")
        self.assertTrue(len(verdict) >= 10,
                        f"Narrative verdict too short: '{verdict}' ({len(verdict)} chars)")

    def test_guide_observation_nonempty(self):
        """guide_observation ≥10 chars when summary exists."""
        from agents_api.dsw.standardize import standardize_output
        result = standardize_output(
            {"summary": "Process capability Cpk = 1.33. Process is capable."},
            "spc", "capability",
        )
        obs = result.get("guide_observation", "")
        self.assertTrue(len(obs) >= 10,
                        f"guide_observation too short: '{obs}' ({len(obs)} chars)")


class EducationTest(SimpleTestCase):
    """DSW-001 §8.4: Education content completeness and depth."""

    def test_all_registered_analyses_have_education(self):
        """Every registered analysis has education content."""
        from agents_api.dsw.registry import ANALYSIS_REGISTRY
        from agents_api.dsw.education import get_education
        missing = []
        for (atype, aid) in ANALYSIS_REGISTRY:
            edu = get_education(atype, aid)
            if not edu:
                missing.append(f"{atype}/{aid}")
        self.assertEqual(len(missing), 0,
                         f"{len(missing)} analyses missing education: {missing[:10]}")

    def test_education_has_required_fields(self):
        """Education entries have title and content."""
        from agents_api.dsw.education import EDUCATION_CONTENT
        for key, entry in EDUCATION_CONTENT.items():
            self.assertIn("title", entry, f"Education {key} missing 'title'")
            self.assertIn("content", entry, f"Education {key} missing 'content'")
            self.assertTrue(len(entry["content"]) > 10,
                            f"Education {key} has empty content")

    def test_education_content_depth(self):
        """Education content ≥200 characters (not shallow stubs)."""
        from agents_api.dsw.education import EDUCATION_CONTENT
        shallow = []
        for key, entry in EDUCATION_CONTENT.items():
            content = entry.get("content", "")
            if len(content) < 200:
                shallow.append(f"{key}: {len(content)} chars")
        self.assertEqual(len(shallow), 0,
                         f"{len(shallow)} entries below 200-char minimum: {shallow[:5]}")

    def test_education_has_dl_structure(self):
        """Education content uses <dl> definition lists."""
        from agents_api.dsw.education import EDUCATION_CONTENT
        no_dl = []
        for key, entry in EDUCATION_CONTENT.items():
            content = entry.get("content", "")
            if "<dl>" not in content or "<dt>" not in content:
                no_dl.append(str(key))
        self.assertEqual(len(no_dl), 0,
                         f"{len(no_dl)} entries missing <dl>/<dt> structure: {no_dl[:5]}")

    def test_education_title_length(self):
        """Education titles ≥15 characters."""
        from agents_api.dsw.education import EDUCATION_CONTENT
        short = []
        for key, entry in EDUCATION_CONTENT.items():
            title = entry.get("title", "")
            if len(title) < 15:
                short.append(f"{key}: '{title}'")
        self.assertEqual(len(short), 0,
                         f"{len(short)} titles too short: {short[:5]}")


class ChartDefaultsTest(SimpleTestCase):
    """DSW-001 §8.5: Chart standardization."""

    def test_height_standardized(self):
        """apply_chart_defaults sets standard height."""
        from agents_api.dsw.chart_defaults import apply_chart_defaults, CHART_HEIGHT
        plot = {"data": [], "layout": {}}
        apply_chart_defaults(plot)
        self.assertEqual(plot["layout"]["height"], CHART_HEIGHT)

    def test_legend_placement(self):
        """Legend defaults to bottom-left horizontal."""
        from agents_api.dsw.chart_defaults import apply_chart_defaults
        plot = {"data": [], "layout": {}}
        apply_chart_defaults(plot)
        self.assertEqual(plot["layout"]["legend"]["orientation"], "h")

    def test_colors_from_palette(self):
        """Trace colors come from SVEND_COLORS palette."""
        from agents_api.dsw.chart_defaults import apply_chart_defaults
        from agents_api.dsw.common import SVEND_COLORS
        plot = {"data": [{"type": "bar", "x": [1], "y": [1]}], "layout": {}}
        apply_chart_defaults(plot)
        self.assertEqual(plot["data"][0]["marker"]["color"], SVEND_COLORS[0])

    def test_transparent_background(self):
        """Charts have transparent background for theme compatibility."""
        from agents_api.dsw.chart_defaults import apply_chart_defaults
        plot = {"data": [], "layout": {}}
        apply_chart_defaults(plot)
        self.assertEqual(plot["layout"]["paper_bgcolor"], "rgba(0,0,0,0)")
        self.assertEqual(plot["layout"]["plot_bgcolor"], "rgba(0,0,0,0)")

    def test_margins_applied(self):
        """Charts have standard margins."""
        from agents_api.dsw.chart_defaults import apply_chart_defaults, CHART_MARGINS
        plot = {"data": [], "layout": {}}
        apply_chart_defaults(plot)
        for key, val in CHART_MARGINS.items():
            self.assertEqual(plot["layout"]["margin"][key], val,
                             f"Margin {key} should be {val}")

    def test_real_analysis_charts_styled(self):
        """Charts from a real analysis have defaults applied after standardization."""
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        df = pd.DataFrame({'measurement': np.random.normal(50, 2, 100)})
        result = _run_analysis('spc', 'capability', df,
                               {'column': 'measurement', 'lsl': 44, 'usl': 56})
        plots = result.get("plots", [])
        for i, plot in enumerate(plots):
            if isinstance(plot, dict) and "layout" in plot:
                layout = plot["layout"]
                self.assertIn("height", layout, f"Plot {i} missing height")
                self.assertEqual(layout.get("paper_bgcolor"), "rgba(0,0,0,0)",
                                 f"Plot {i} has non-transparent background")


class NewStatisticsTest(SimpleTestCase):
    """DSW-001 §8.6: Evidence grade and Bayesian shadow."""

    def test_pvalue_analyses_have_evidence_grade(self):
        """Post-processor generates evidence_grade when p_value present."""
        from agents_api.dsw.standardize import standardize_output
        result = standardize_output(
            {"summary": "t", "p_value": 0.01,
             "statistics": {"p_value": 0.01, "effect_size_r": 0.4, "n": 50}},
            "stats", "mann_whitney",
        )
        self.assertIsNotNone(result.get("evidence_grade"),
                             "evidence_grade not generated for p-value result")

    def test_pvalue_analyses_have_bayesian_shadow(self):
        """Post-processor generates bayesian_shadow for shadow-eligible analyses with sufficient stats."""
        from agents_api.dsw.standardize import standardize_output
        result = standardize_output(
            {"summary": "t", "p_value": 0.01,
             "statistics": {"p_value": 0.01, "effect_size_r": 0.4, "n": 50}},
            "stats", "mann_whitney",
        )
        self.assertIsNotNone(result.get("bayesian_shadow"),
                             "bayesian_shadow not generated for nonparametric analysis with effect_r + n")


class WhatIfTest(SimpleTestCase):
    """DSW-001 §8.7: What-if interactivity."""

    def test_tier1_analyses_have_whatif(self):
        """Post-processor creates what_if stub for tier 1 analyses."""
        from agents_api.dsw.standardize import standardize_output
        result = standardize_output({"summary": "t"}, "stats", "power_z")
        self.assertIsNotNone(result.get("what_if"),
                             "what_if not generated for tier 1 analysis")

    def test_whatif_schema_valid(self):
        """what_if dict has required schema fields."""
        from agents_api.dsw.standardize import standardize_output
        result = standardize_output(
            {"summary": "t", "power_explorer": {
                "test_type": "ttest", "observed_effect": 1.0,
                "observed_std": 1.0, "observed_n": 30,
                "alpha": 0.05, "cohens_d": 0.5,
            }},
            "stats", "regression",
        )
        wi = result.get("what_if")
        self.assertIsNotNone(wi)
        self.assertIn("type", wi)
        self.assertIn("parameters", wi)
        self.assertIn("endpoint", wi)


class FrontendRenderingTest(SimpleTestCase):
    """DSW-001 §8.8: Frontend rendering elements."""

    def _template(self):
        return _read(WEB_ROOT / "templates" / "analysis_workbench.html")

    def test_education_details_element(self):
        """Frontend renders education as collapsible <details> panel."""
        t = self._template()
        self.assertIn("dsw-education", t)
        self.assertIn("dsw-panel", t)
        self.assertIn("details.className = 'dsw-panel dsw-education'", t)

    def test_narrative_css_classes(self):
        """Frontend has all narrative CSS classes."""
        t = self._template()
        for cls in ["dsw-narrative", "dsw-verdict", "dsw-narrative-body",
                     "dsw-next", "dsw-chart-guidance"]:
            self.assertIn(cls, t, f"Missing CSS class: {cls}")

    def test_evidence_badge_css_classes(self):
        """Frontend has evidence grade badge CSS for all levels."""
        t = self._template()
        self.assertIn("dsw-evidence-badge", t)
        for level in ["strong", "moderate", "weak", "inconclusive"]:
            self.assertIn(f".dsw-evidence-badge.{level}", t,
                          f"Missing badge CSS for level: {level}")

    def test_bayesian_panel_css(self):
        """Frontend has Bayesian shadow panel CSS."""
        t = self._template()
        self.assertIn("dsw-bayesian-panel", t)

    def test_diagnostics_panel_css(self):
        """Frontend has diagnostics panel CSS."""
        t = self._template()
        self.assertIn("dsw-diagnostics", t)
        for cls in ["dsw-diag-pass", "dsw-diag-warn", "dsw-diag-fail"]:
            self.assertIn(cls, t, f"Missing diagnostic CSS: {cls}")

    def test_what_if_slider_css(self):
        """Frontend has what-if slider CSS."""
        t = self._template()
        self.assertIn("dsw-what-if", t)
        self.assertIn("dsw-slider", t)

    def test_render_function_called(self):
        """renderDSWBlocks(result) is called in the response handler."""
        t = self._template()
        self.assertIn("renderDSWBlocks(result)", t)
