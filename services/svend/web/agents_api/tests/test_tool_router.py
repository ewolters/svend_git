"""
ToolRouter behavioral tests.

Standard:     ARCH-001 §3 (Module Registration)
Compliance:   TST-001 §4

<!-- test: agents_api.tests.test_tool_router.ToolRouterRegistrationTests -->
<!-- test: agents_api.tests.test_tool_router.ToolRouterURLTests -->
<!-- test: agents_api.tests.test_tool_router.ToolRouterPermissionTests -->

CR: 94a4be9b
"""

from django.http import JsonResponse
from django.test import TestCase, override_settings

from agents_api.tool_router import ToolRouter

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


# =============================================================================
# Helpers — mock views and models
# =============================================================================


def _make_view(name="mock"):
    """Return a simple view function with a distinguishable name."""

    def view(request, *args, **kwargs):
        return JsonResponse({"view": name})

    view.__name__ = name
    view.__qualname__ = name
    return view


def _mock_model():
    """Return a mock model class."""
    m = type("MockModel", (), {"__name__": "MockModel"})
    return m


def _register_sample(slug="widget", **overrides):
    """Register a sample tool with sensible defaults.  Returns the slug."""
    defaults = {
        "slug": slug,
        "model": _mock_model(),
        "list_view": _make_view("list"),
        "create_view": _make_view("create"),
        "detail_view": _make_view("detail"),
        "update_view": _make_view("update"),
        "delete_view": _make_view("delete"),
    }
    defaults.update(overrides)
    ToolRouter.register(**defaults)
    return slug


# =============================================================================
# Registration tests
# =============================================================================


@SECURE_OFF
class ToolRouterRegistrationTests(TestCase):
    """Verify tool registration stores config and enforces uniqueness."""

    def setUp(self):
        ToolRouter.clear()

    def tearDown(self):
        ToolRouter.clear()

    def test_register_stores_config(self):
        """Registered tool config is retrievable by slug."""
        _register_sample("a3")
        cfg = ToolRouter.get_tool("a3")
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["slug"], "a3")
        self.assertEqual(cfg["permission"], "paid")  # default

    def test_duplicate_slug_raises(self):
        """Registering the same slug twice raises ValueError."""
        _register_sample("fmea")
        with self.assertRaises(ValueError) as ctx:
            _register_sample("fmea")
        self.assertIn("already registered", str(ctx.exception))

    def test_list_tools_returns_all(self):
        """list_tools() returns every registered tool."""
        _register_sample("a3")
        _register_sample("fmea")
        _register_sample("rca")
        tools = ToolRouter.list_tools()
        slugs = {t["slug"] for t in tools}
        self.assertEqual(slugs, {"a3", "fmea", "rca"})

    def test_get_tool_missing_returns_none(self):
        """get_tool() returns None for unregistered slug."""
        self.assertIsNone(ToolRouter.get_tool("nonexistent"))

    def test_invalid_permission_raises(self):
        """Unknown permission level raises ValueError at registration time."""
        with self.assertRaises(ValueError) as ctx:
            _register_sample("bad", permission="platinum")
        self.assertIn("Unknown permission", str(ctx.exception))

    def test_config_stores_actions(self):
        """Actions and collection_actions are stored in the config."""
        action_view = _make_view("act")
        col_view = _make_view("col")
        _register_sample(
            "tool",
            actions={"do-thing": action_view},
            collection_actions={"summary": col_view},
        )
        cfg = ToolRouter.get_tool("tool")
        self.assertIn("do-thing", cfg["actions"])
        self.assertIn("summary", cfg["collection_actions"])


# =============================================================================
# URL pattern tests
# =============================================================================


@SECURE_OFF
class ToolRouterURLTests(TestCase):
    """Verify generated URL patterns are correct and resolvable."""

    def setUp(self):
        ToolRouter.clear()

    def tearDown(self):
        ToolRouter.clear()

    def test_crud_patterns_generated(self):
        """Standard CRUD patterns are generated for a registered tool."""
        _register_sample("a3")
        patterns = ToolRouter.get_urlpatterns()
        routes = {p.pattern._route for p in patterns}
        self.assertIn("a3/", routes)
        self.assertIn("a3/create/", routes)
        self.assertIn("a3/<uuid:pk>/", routes)
        self.assertIn("a3/<uuid:pk>/update/", routes)
        self.assertIn("a3/<uuid:pk>/delete/", routes)

    def test_crud_url_names(self):
        """URL names follow the {slug}-{operation} convention."""
        _register_sample("fmea")
        patterns = ToolRouter.get_urlpatterns()
        names = {p.name for p in patterns}
        for suffix in ("list", "create", "detail", "update", "delete"):
            self.assertIn(f"fmea-{suffix}", names)

    def test_action_patterns(self):
        """Instance-level actions mount at <uuid:pk>/{action}/."""
        _register_sample(
            "a3",
            actions={
                "auto-populate": _make_view("autopop"),
                "critique": _make_view("critique"),
            },
        )
        patterns = ToolRouter.get_urlpatterns()
        routes = {p.pattern._route for p in patterns}
        self.assertIn("a3/<uuid:pk>/auto-populate/", routes)
        self.assertIn("a3/<uuid:pk>/critique/", routes)

    def test_collection_action_patterns(self):
        """Collection-level actions mount at {slug}/{action}/."""
        _register_sample(
            "fmea",
            collection_actions={"patterns": _make_view("patterns")},
        )
        patterns = ToolRouter.get_urlpatterns()
        routes = {p.pattern._route for p in patterns}
        self.assertIn("fmea/patterns/", routes)

    def test_action_url_names_with_slashes(self):
        """Actions with slashes in the name get dashes in URL names."""
        _register_sample(
            "a3",
            actions={"export/pdf": _make_view("export_pdf")},
        )
        patterns = ToolRouter.get_urlpatterns()
        names = {p.name for p in patterns}
        self.assertIn("a3-export-pdf", names)

    def test_nested_resource_patterns(self):
        """Nested resources mount list and detail patterns."""
        _register_sample(
            "fmea",
            nested_resources=[
                {
                    "name": "rows",
                    "list_view": _make_view("row_list"),
                    "detail_view": _make_view("row_detail"),
                },
            ],
        )
        patterns = ToolRouter.get_urlpatterns()
        routes = {p.pattern._route for p in patterns}
        self.assertIn("fmea/<uuid:pk>/rows/", routes)
        self.assertIn("fmea/<uuid:pk>/rows/<uuid:item_id>/", routes)

    def test_multiple_tools_combined(self):
        """get_urlpatterns() includes patterns from all registered tools."""
        _register_sample("a3")
        _register_sample("fmea")
        patterns = ToolRouter.get_urlpatterns()
        routes = {p.pattern._route for p in patterns}
        self.assertIn("a3/", routes)
        self.assertIn("fmea/", routes)
        # Each tool should have 5 CRUD patterns minimum
        a3_routes = [r for r in routes if r.startswith("a3")]
        fmea_routes = [r for r in routes if r.startswith("fmea")]
        self.assertGreaterEqual(len(a3_routes), 5)
        self.assertGreaterEqual(len(fmea_routes), 5)

    def test_empty_registry_returns_empty_list(self):
        """get_urlpatterns() returns empty list when nothing is registered."""
        self.assertEqual(ToolRouter.get_urlpatterns(), [])


# =============================================================================
# Permission decorator tests
# =============================================================================


@SECURE_OFF
class ToolRouterPermissionTests(TestCase):
    """Verify permission decorators are applied to generated views."""

    def setUp(self):
        ToolRouter.clear()

    def tearDown(self):
        ToolRouter.clear()

    def test_paid_wraps_views(self):
        """Permission 'paid' wraps views with gated_paid decorator."""
        list_view = _make_view("list")
        _register_sample("a3", list_view=list_view, permission="paid")
        patterns = ToolRouter.get_urlpatterns()
        list_pattern = next(p for p in patterns if p.name == "a3-list")
        # The wrapped view should NOT be the original function
        self.assertIsNot(list_pattern.callback, list_view)
        # The wrapper's __wrapped__ should point to the original
        self.assertIs(list_pattern.callback.__wrapped__, list_view)

    def test_free_does_not_wrap(self):
        """Permission 'free' leaves views unwrapped."""
        list_view = _make_view("list")
        _register_sample("a3", list_view=list_view, permission="free")
        patterns = ToolRouter.get_urlpatterns()
        list_pattern = next(p for p in patterns if p.name == "a3-list")
        self.assertIs(list_pattern.callback, list_view)

    def test_team_wraps_views(self):
        """Permission 'team' wraps views with require_team decorator."""
        detail_view = _make_view("detail")
        _register_sample("hoshin", detail_view=detail_view, permission="team")
        patterns = ToolRouter.get_urlpatterns()
        detail_pattern = next(p for p in patterns if p.name == "hoshin-detail")
        self.assertIsNot(detail_pattern.callback, detail_view)
        self.assertIs(detail_pattern.callback.__wrapped__, detail_view)

    def test_enterprise_wraps_views(self):
        """Permission 'enterprise' wraps views with require_enterprise."""
        create_view = _make_view("create")
        _register_sample("advanced", create_view=create_view, permission="enterprise")
        patterns = ToolRouter.get_urlpatterns()
        create_pattern = next(p for p in patterns if p.name == "advanced-create")
        self.assertIsNot(create_pattern.callback, create_view)
        self.assertIs(create_pattern.callback.__wrapped__, create_view)

    def test_actions_also_wrapped(self):
        """Permission decorator is applied to action views too."""
        action_view = _make_view("autopop")
        _register_sample(
            "a3",
            permission="paid",
            actions={"auto-populate": action_view},
        )
        patterns = ToolRouter.get_urlpatterns()
        action_pattern = next(p for p in patterns if p.name == "a3-auto-populate")
        self.assertIsNot(action_pattern.callback, action_view)
        self.assertIs(action_pattern.callback.__wrapped__, action_view)


# =============================================================================
# Clear / reset tests
# =============================================================================


@SECURE_OFF
class ToolRouterClearTests(TestCase):
    """Verify clear() resets the registry completely."""

    def setUp(self):
        ToolRouter.clear()

    def tearDown(self):
        ToolRouter.clear()

    def test_clear_empties_registry(self):
        """clear() removes all registered tools."""
        _register_sample("a3")
        _register_sample("fmea")
        self.assertEqual(len(ToolRouter.list_tools()), 2)
        ToolRouter.clear()
        self.assertEqual(len(ToolRouter.list_tools()), 0)
        self.assertIsNone(ToolRouter.get_tool("a3"))

    def test_reregister_after_clear(self):
        """A slug can be re-registered after clear()."""
        _register_sample("a3")
        ToolRouter.clear()
        _register_sample("a3")  # should not raise
        self.assertIsNotNone(ToolRouter.get_tool("a3"))

    def test_clear_does_not_affect_new_registrations(self):
        """Tools registered after clear() work normally."""
        _register_sample("old")
        ToolRouter.clear()
        _register_sample("new")
        patterns = ToolRouter.get_urlpatterns()
        routes = {p.pattern._route for p in patterns}
        self.assertNotIn("old/", routes)
        self.assertIn("new/", routes)
