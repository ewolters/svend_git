"""
Central URL router for pluggable QMS tool modules.

Standard:     ARCH-001 §3 (Module Registration)
Compliance:   ORG-001 §2.2, SEC-001 §5.2
"""

from django.urls import path

from accounts.permissions import gated_paid, require_enterprise, require_team

# ---------------------------------------------------------------------------
# Permission level → decorator mapping
# ---------------------------------------------------------------------------

_PERMISSION_DECORATORS = {
    "paid": gated_paid,
    "team": require_team,
    "enterprise": require_enterprise,
    "free": None,
}


class ToolRouter:
    """Class-based registry that auto-generates Django URL patterns for QMS
    tool modules.

    Usage::

        ToolRouter.register(
            slug="a3",
            model=A3Report,
            list_view=list_a3_reports,
            create_view=create_a3_report,
            detail_view=get_a3_report,
            update_view=update_a3_report,
            delete_view=delete_a3_report,
            permission="paid",
            actions={"auto-populate": auto_populate_a3},
            collection_actions={"patterns": cross_fmea_patterns},
        )

        urlpatterns = ToolRouter.get_urlpatterns()
    """

    # Class-level storage — singleton pattern
    _registry: dict = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        *,
        slug: str,
        model,
        list_view,
        create_view,
        detail_view,
        update_view,
        delete_view,
        permission: str = "paid",
        actions: dict | None = None,
        collection_actions: dict | None = None,
        nested_resources: list | None = None,
        path_prefix: str = "",
        pk_name: str = "pk",
    ) -> None:
        """Register a QMS tool module.

        Args:
            slug:               URL prefix, e.g. ``"a3"`` or ``"fmea"``.
            model:              Django model class for the tool.
            list_view:          View for listing resources.
            create_view:        View for creating a resource.
            detail_view:        View for retrieving a single resource.
            update_view:        View for updating a resource.
            delete_view:        View for deleting a resource.
            permission:         ``"paid"`` | ``"team"`` | ``"enterprise"`` | ``"free"``.
            actions:            Instance-level action views keyed by URL suffix.
            collection_actions: Collection-level action views keyed by URL suffix.
            nested_resources:   List of nested resource names.
            path_prefix:        Extra path prefix after slug, e.g. ``"sessions"``
                                for legacy ``{slug}/sessions/`` URL structure.
            pk_name:            URL parameter name for primary key, default ``"pk"``.
                                Use for legacy patterns like ``"diagram_id"``.
        """
        if slug in cls._registry:
            raise ValueError(f"Tool slug '{slug}' is already registered")

        if permission not in _PERMISSION_DECORATORS:
            raise ValueError(f"Unknown permission level '{permission}'. Valid: {', '.join(_PERMISSION_DECORATORS)}")

        cls._registry[slug] = {
            "slug": slug,
            "model": model,
            "list_view": list_view,
            "create_view": create_view,
            "detail_view": detail_view,
            "update_view": update_view,
            "delete_view": delete_view,
            "permission": permission,
            "actions": actions or {},
            "collection_actions": collection_actions or {},
            "nested_resources": nested_resources or [],
            "path_prefix": path_prefix,
            "pk_name": pk_name,
        }

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    @classmethod
    def get_tool(cls, slug: str) -> dict | None:
        """Return the config dict for *slug*, or ``None``."""
        return cls._registry.get(slug)

    @classmethod
    def list_tools(cls) -> list[dict]:
        """Return all registered tool configs."""
        return list(cls._registry.values())

    # ------------------------------------------------------------------
    # URL generation
    # ------------------------------------------------------------------

    @classmethod
    def get_urlpatterns(cls) -> list:
        """Build and return Django URL patterns for every registered tool."""
        patterns = []
        for slug, cfg in cls._registry.items():
            patterns.extend(cls._patterns_for_tool(slug, cfg))
        return patterns

    @classmethod
    def _wrap(cls, view, permission: str):
        """Apply the permission decorator for *permission* to *view*."""
        decorator = _PERMISSION_DECORATORS.get(permission)
        if decorator is None:
            return view
        return decorator(view)

    @classmethod
    def _patterns_for_tool(cls, slug: str, cfg: dict) -> list:
        perm = cfg["permission"]
        prefix = cfg.get("path_prefix", "")
        pk = cfg.get("pk_name", "pk")
        # Build base path: "{slug}/" or "{slug}/{prefix}/" if prefix set
        base = f"{slug}/{prefix}/" if prefix else f"{slug}/"
        item = f"{base}<uuid:{pk}>/"
        patterns = [
            path(base, cls._wrap(cfg["list_view"], perm), name=f"{slug}-list"),
            path(f"{base}create/", cls._wrap(cfg["create_view"], perm), name=f"{slug}-create"),
            path(item, cls._wrap(cfg["detail_view"], perm), name=f"{slug}-detail"),
            path(f"{item}update/", cls._wrap(cfg["update_view"], perm), name=f"{slug}-update"),
            path(f"{item}delete/", cls._wrap(cfg["delete_view"], perm), name=f"{slug}-delete"),
        ]

        # Instance-level actions: {item}{action_name}/
        for action_name, action_view in cfg["actions"].items():
            patterns.append(
                path(
                    f"{item}{action_name}/",
                    cls._wrap(action_view, perm),
                    name=f"{slug}-{action_name.replace('/', '-')}",
                )
            )

        # Collection-level actions: {slug}/{action_name}/
        # Mounted at slug level (not base/prefix level) so they're accessible
        # without the path_prefix. E.g. rca/critique/ not rca/sessions/critique/.
        slug_base = f"{slug}/"
        for action_name, action_view in cfg["collection_actions"].items():
            patterns.append(
                path(
                    f"{slug_base}{action_name}/",
                    cls._wrap(action_view, perm),
                    name=f"{slug}-{action_name.replace('/', '-')}",
                )
            )

        # Nested resources: {item}{resource}/ and {item}{resource}/<uuid:item_id>/
        for nested in cfg["nested_resources"]:
            if isinstance(nested, str):
                resource_name = nested
                nested_list_view = None
                nested_detail_view = None
            elif isinstance(nested, dict):
                resource_name = nested["name"]
                nested_list_view = nested.get("list_view")
                nested_detail_view = nested.get("detail_view")
            else:
                continue

            if nested_list_view:
                patterns.append(
                    path(
                        f"{item}{resource_name}/",
                        cls._wrap(nested_list_view, perm),
                        name=f"{slug}-{resource_name}-list",
                    )
                )
            if nested_detail_view:
                patterns.append(
                    path(
                        f"{item}{resource_name}/<uuid:item_id>/",
                        cls._wrap(nested_detail_view, perm),
                        name=f"{slug}-{resource_name}-detail",
                    )
                )

        return patterns

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    @classmethod
    def clear(cls) -> None:
        """Remove all registered tools.  Intended for test tearDown."""
        cls._registry = {}
