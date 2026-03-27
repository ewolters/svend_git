"""
Central tool registration for QMS modules using ToolRouter.

Import this module to register all tools. Called from agents_api.apps.ready().

Standard:     ARCH-001 §10.1 (ToolRouter)
Compliance:   ORG-001 §2.2

CR: 722e1cb1
"""

from .models import CEMatrix, IshikawaDiagram
from .tool_router import ToolRouter

# Lazy imports to avoid circular dependencies at module level
_registered = False


def register_tools():
    """Register all QMS tools with the ToolRouter.

    Safe to call multiple times — skips if already registered.
    """
    global _registered
    if _registered:
        return
    _registered = True

    from . import ce_views, ishikawa_views

    # ------------------------------------------------------------------
    # Ishikawa (Fishbone) Diagrams
    # URL: /api/ishikawa/sessions/, /api/ishikawa/sessions/<uuid>/...
    # ------------------------------------------------------------------
    ToolRouter.register(
        slug="ishikawa",
        model=IshikawaDiagram,
        list_view=ishikawa_views.list_diagrams,
        create_view=ishikawa_views.create_diagram,
        detail_view=ishikawa_views.get_diagram,
        update_view=ishikawa_views.update_diagram,
        delete_view=ishikawa_views.delete_diagram,
        permission="paid",
        path_prefix="sessions",
        pk_name="diagram_id",
    )

    # ------------------------------------------------------------------
    # Cause & Effect (C&E) Matrix
    # URL: /api/ce/sessions/, /api/ce/sessions/<uuid>/...
    # ------------------------------------------------------------------
    ToolRouter.register(
        slug="ce",
        model=CEMatrix,
        list_view=ce_views.list_matrices,
        create_view=ce_views.create_matrix,
        detail_view=ce_views.get_matrix,
        update_view=ce_views.update_matrix,
        delete_view=ce_views.delete_matrix,
        permission="paid",
        path_prefix="sessions",
        pk_name="matrix_id",
    )
