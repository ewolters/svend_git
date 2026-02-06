"""SPC (Statistical Process Control) API URLs."""

from django.urls import path

from . import spc_views as views

urlpatterns = [
    # File Upload & Analysis
    path("upload/", views.upload_data, name="spc_upload"),
    path("analyze/", views.analyze_uploaded, name="spc_analyze"),

    # Control Charts (direct data)
    path("chart/", views.control_chart, name="spc_control_chart"),
    path("chart/recommend/", views.recommend_chart, name="spc_recommend_chart"),
    path("chart/types/", views.chart_types, name="spc_chart_types"),

    # Process Capability
    path("capability/", views.capability_study, name="spc_capability"),

    # Statistical Summary
    path("summary/", views.statistical_summary, name="spc_summary"),
]
