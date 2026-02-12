"""Guide API URLs."""

from django.urls import path
from . import guide_views

urlpatterns = [
    path("chat/", guide_views.guide_chat, name="guide_chat"),
    path("summarize/", guide_views.summarize_project, name="guide_summarize"),
    path("rate-limit/", guide_views.rate_limit_status, name="guide_rate_limit"),
]
