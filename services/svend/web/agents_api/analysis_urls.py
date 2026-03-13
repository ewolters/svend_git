"""URL routing for analysis/ dispatch — independent of dsw/.

CR: b3b2bf39
"""

from django.urls import path

from .analysis.dispatch import run_analysis

urlpatterns = [
    path("run/", run_analysis, name="analysis_run"),
]
