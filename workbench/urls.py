"""URL routes for Analysis Workbench API.

Session-based persistence for the analysis workbench.
Pull contract endpoints for cross-tool integration.
"""

from django.urls import path

from . import views

app_name = "workbench"

urlpatterns = [
    # Sessions
    path("sessions/", views.session_list_create, name="session_list_create"),
    path("sessions/<uuid:session_id>/", views.session_detail, name="session_detail"),
    # Datasets within a session
    path("sessions/<uuid:session_id>/datasets/", views.dataset_list_create, name="dataset_list_create"),
    path("sessions/<uuid:session_id>/datasets/<uuid:dataset_id>/", views.dataset_detail, name="dataset_detail"),
    # Analyses within a session
    path("sessions/<uuid:session_id>/analyses/", views.analysis_list_create, name="analysis_list_create"),
    path("sessions/<uuid:session_id>/analyses/<uuid:analysis_id>/", views.analysis_detail, name="analysis_detail"),
    # Pull contract — manifest + sub-artifact access
    path("sessions/<uuid:session_id>/manifest/", views.session_manifest, name="session_manifest"),
    path("sessions/<uuid:session_id>/references/", views.session_references, name="session_references"),
    path(
        "sessions/<uuid:session_id>/analyses/<uuid:analysis_id>/<path:key_path>/",
        views.analysis_sub_artifact,
        name="analysis_sub_artifact",
    ),
    # Pull contract — direct analysis access + reference registration
    path("analyses/<uuid:analysis_id>/", views.analysis_pull_detail, name="analysis_pull_detail"),
    path("analyses/<uuid:analysis_id>/references/", views.analysis_register_reference, name="analysis_register_ref"),
    path(
        "analyses/<uuid:analysis_id>/<path:key_path>/",
        views.analysis_pull_sub_artifact,
        name="analysis_pull_sub_artifact",
    ),
]
