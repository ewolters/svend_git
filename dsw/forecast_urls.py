"""Forecast API URLs."""

from django.urls import path

from . import forecast_views as views

urlpatterns = [
    path("", views.forecast, name="forecast"),
    path("quote/", views.quote, name="quote"),
]
