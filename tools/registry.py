"""
Central tool registration for QMS modules using ToolRouter.

Import this module to register all tools. Called from svend/urls.py.

Standard:     ARCH-001 §10.1 (ToolRouter)
Compliance:   ORG-001 §2.2

Transitional: During extraction, tool registrations migrate from here
to each tool app's AppConfig.ready(). By Phase 3 cutover, this file
should contain only ForgeRack (if still in agents_api).
"""

from .router import ToolRouter

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

    from agents_api import a3_views, ce_views, ishikawa_views, rca_views
    from agents_api.models import A3Report, CEMatrix, IshikawaDiagram, RCASession

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

    # ------------------------------------------------------------------
    # A3 Problem-Solving Reports
    # URL: /api/a3/, /api/a3/<uuid>/...
    # ------------------------------------------------------------------
    ToolRouter.register(
        slug="a3",
        model=A3Report,
        list_view=a3_views.list_a3_reports,
        create_view=a3_views.create_a3_report,
        detail_view=a3_views.get_a3_report,
        update_view=a3_views.update_a3_report,
        delete_view=a3_views.delete_a3_report,
        permission="paid",
        pk_name="report_id",
        actions={
            "import": a3_views.import_to_a3,
            "auto-populate": a3_views.auto_populate_a3,
            "critique": a3_views.critique_a3,
            "embed-diagram": a3_views.embed_diagram,
            "export/pdf": a3_views.export_a3_pdf,
            "actions": a3_views.list_a3_actions,
            "actions/create": a3_views.create_a3_action,
        },
    )

    # VSM — extracted to vsm/ app. URLs at /api/vsm/ via vsm.urls.

    # ------------------------------------------------------------------
    # Root Cause Analysis
    # URL: /api/rca/sessions/, /api/rca/sessions/<uuid>/...
    # Collection actions at /api/rca/ for stateless AI operations.
    # ------------------------------------------------------------------
    ToolRouter.register(
        slug="rca",
        model=RCASession,
        list_view=rca_views.list_sessions,
        create_view=rca_views.create_session,
        detail_view=rca_views.get_session,
        update_view=rca_views.update_session,
        delete_view=rca_views.delete_session,
        permission="paid",
        path_prefix="sessions",
        pk_name="session_id",
        actions={
            "link-a3": rca_views.link_to_a3,
            "actions": rca_views.list_rca_actions,
            "actions/create": rca_views.create_rca_action,
        },
        collection_actions={
            "critique": rca_views.critique,
            "critique-countermeasure": rca_views.critique_countermeasure,
            "evaluate": rca_views.evaluate_chain,
            "guided-questions": rca_views.guided_questions,
            "clusters": rca_views.cluster_root_causes,
            "similar": rca_views.find_similar,
            "reindex": rca_views.reindex_embeddings,
        },
    )

    # ------------------------------------------------------------------
    # ForgeRack Sessions
    # URL: /api/rack/sessions/, /api/rack/sessions/<uuid>/...
    # ------------------------------------------------------------------
    from agents_api import rack_views
    from agents_api.models import RackSession

    ToolRouter.register(
        slug="rack",
        model=RackSession,
        list_view=rack_views.list_rack_sessions,
        create_view=rack_views.create_rack_session,
        detail_view=rack_views.get_rack_session,
        update_view=rack_views.update_rack_session,
        delete_view=rack_views.delete_rack_session,
        permission="free",
        path_prefix="sessions",
        pk_name="session_id",
    )


def register_tool(slug, **kwargs):
    """Public API for self-registration by tool apps.

    Called from each tool's AppConfig.ready() during Phase 1A+ extractions.
    """
    ToolRouter.register(slug=slug, **kwargs)
