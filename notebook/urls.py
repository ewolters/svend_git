"""URL routes for Notebook API — NB-001."""

from django.urls import path

from agents_api import a3_views

from . import views

urlpatterns = [
    # Notebooks
    path("", views.list_create_notebooks, name="notebook_list_create"),
    path("<uuid:notebook_id>/", views.notebook_detail, name="notebook_detail"),
    path(
        "<uuid:notebook_id>/conclude/",
        views.conclude_notebook,
        name="notebook_conclude",
    ),
    # Trials
    path(
        "<uuid:notebook_id>/trials/",
        views.list_create_trials,
        name="trial_list_create",
    ),
    path(
        "<uuid:notebook_id>/trials/<uuid:trial_id>/",
        views.trial_detail,
        name="trial_detail",
    ),
    path(
        "<uuid:notebook_id>/trials/<uuid:trial_id>/complete/",
        views.complete_trial,
        name="trial_complete",
    ),
    # Pages
    path(
        "<uuid:notebook_id>/pages/",
        views.list_create_pages,
        name="page_list_create",
    ),
    # Pull tool outputs into notebook
    path("<uuid:notebook_id>/pull/", views.pull_tool, name="notebook_pull_tool"),
    # Front Page (aggregated knowledge base)
    path("front-page/", views.front_page, name="front_page"),
    # Front Matter (per-notebook)
    path(
        "<uuid:notebook_id>/front-matter/",
        views.add_front_matter,
        name="front_matter_add",
    ),
    # Projections — notebook → output formats
    path(
        "<uuid:notebook_id>/project/a3/",
        a3_views.project_notebook_to_a3,
        name="notebook_project_a3",
    ),
    # Yokoten
    path("yokoten/", views.list_yokoten, name="yokoten_list"),
    path(
        "yokoten/<uuid:yokoten_id>/adopt/",
        views.adopt_yokoten,
        name="yokoten_adopt",
    ),
]
