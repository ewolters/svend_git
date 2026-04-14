"""Guide API URLs."""

from django.urls import path

from . import views

urlpatterns = [
    path("chat/", views.guide_chat, name="guide_chat"),
    path("summarize/", views.summarize_project, name="guide_summarize"),
    path("rate-limit/", views.rate_limit_status, name="guide_rate_limit"),
]
