"""Learning module URL configuration."""

from django.urls import path

from . import learn_views

app_name = "learn"

urlpatterns = [
    # Course structure
    path("modules/", learn_views.list_modules, name="modules"),
    path("modules/<str:module_id>/", learn_views.get_module, name="module_detail"),
    path(
        "modules/<str:module_id>/sections/<str:section_id>/",
        learn_views.get_section,
        name="section_detail",
    ),
    # Progress tracking
    path("progress/", learn_views.get_progress, name="progress"),
    path(
        "progress/<str:module_id>/complete/",
        learn_views.mark_section_complete,
        name="complete_section",
    ),
    # Interactive tutorial sessions
    path("session/start/", learn_views.start_session, name="start_session"),
    path(
        "session/<uuid:session_id>/step/<str:step_id>/execute/",
        learn_views.execute_step,
        name="execute_step",
    ),
    path(
        "session/<uuid:session_id>/reset/",
        learn_views.reset_session,
        name="reset_session",
    ),
    # Assessment
    path(
        "assessment/generate/",
        learn_views.generate_assessment,
        name="generate_assessment",
    ),
    path("assessment/submit/", learn_views.submit_assessment, name="submit_assessment"),
    path(
        "assessment/history/", learn_views.assessment_history, name="assessment_history"
    ),
]
