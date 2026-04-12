"""URL routes for Notebook API — NB-001."""

from django.urls import path

from . import a3_views, notebook_views

urlpatterns = [
    # Notebooks
    path("", notebook_views.list_create_notebooks, name="notebook_list_create"),
    path("<uuid:notebook_id>/", notebook_views.notebook_detail, name="notebook_detail"),
    path(
        "<uuid:notebook_id>/conclude/",
        notebook_views.conclude_notebook,
        name="notebook_conclude",
    ),
    # Trials
    path(
        "<uuid:notebook_id>/trials/",
        notebook_views.list_create_trials,
        name="trial_list_create",
    ),
    path(
        "<uuid:notebook_id>/trials/<uuid:trial_id>/",
        notebook_views.trial_detail,
        name="trial_detail",
    ),
    path(
        "<uuid:notebook_id>/trials/<uuid:trial_id>/complete/",
        notebook_views.complete_trial,
        name="trial_complete",
    ),
    # Pages
    path(
        "<uuid:notebook_id>/pages/",
        notebook_views.list_create_pages,
        name="page_list_create",
    ),
    # Pull tool outputs into notebook
    path("<uuid:notebook_id>/pull/", notebook_views.pull_tool, name="notebook_pull_tool"),
    # Front Page (aggregated knowledge base)
    path("front-page/", notebook_views.front_page, name="front_page"),
    # Front Matter (per-notebook)
    path(
        "<uuid:notebook_id>/front-matter/",
        notebook_views.add_front_matter,
        name="front_matter_add",
    ),
    # Projections — notebook → output formats
    path(
        "<uuid:notebook_id>/project/a3/",
        a3_views.project_notebook_to_a3,
        name="notebook_project_a3",
    ),
    # Yokoten
    path("yokoten/", notebook_views.list_yokoten, name="yokoten_list"),
    path(
        "yokoten/<uuid:yokoten_id>/adopt/",
        notebook_views.adopt_yokoten,
        name="yokoten_adopt",
    ),
]
