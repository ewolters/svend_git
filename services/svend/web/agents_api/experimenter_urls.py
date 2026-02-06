"""Experimenter Agent API URLs."""

from django.urls import path

from . import experimenter_views as views

urlpatterns = [
    # Power analysis
    path("power/", views.power_analysis, name="experimenter_power"),

    # Design generation
    path("design/", views.design_experiment, name="experimenter_design"),
    path("design/types/", views.design_types, name="experimenter_design_types"),

    # Full experiment (power + design)
    path("full/", views.full_experiment, name="experimenter_full"),

    # Analysis
    path("analyze/", views.analyze_results, name="experimenter_analyze"),
    path("contour/", views.contour_plot, name="experimenter_contour"),
    path("optimize/", views.optimize_response, name="experimenter_optimize"),

    # DOE Guidance Chat (LLM)
    path("chat/", views.doe_guidance_chat, name="experimenter_chat"),
    path("models/", views.available_models, name="experimenter_models"),
]
