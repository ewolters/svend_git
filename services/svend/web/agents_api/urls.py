"""Agent API URL configuration."""

from django.urls import path

from . import views

app_name = "agents_api"

urlpatterns = [
    path("researcher/", views.researcher_agent, name="researcher"),
    path("coder/", views.coder_agent, name="coder"),
    path("writer/", views.writer_agent, name="writer"),
    path("editor/", views.editor_agent, name="editor"),
    path("experimenter/", views.experimenter_agent, name="experimenter"),
    path("eda/", views.eda_agent, name="eda"),
]
