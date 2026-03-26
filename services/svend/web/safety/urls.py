"""HIRARC Safety URL routes."""

from django.urls import path

from . import views

urlpatterns = [
    # Sites & Auditors
    path("sites/", views.site_list, name="safety_sites"),
    path("auditors/", views.auditor_list, name="safety_auditors"),
    # Frontier Zones
    path("zones/", views.zone_list_create, name="safety_zones"),
    path("zones/<uuid:zone_id>/", views.zone_detail, name="safety_zone_detail"),
    # Audit Scheduling
    path("schedules/", views.schedule_list_create, name="safety_schedules"),
    path("schedules/<uuid:schedule_id>/", views.schedule_detail, name="safety_schedule_detail"),
    path("assignments/<uuid:assignment_id>/", views.assignment_update, name="safety_assignment_update"),
    # Frontier Cards
    path("cards/", views.card_list_create, name="safety_cards"),
    path("cards/<uuid:card_id>/", views.card_detail, name="safety_card_detail"),
    path("cards/<uuid:card_id>/process/", views.process_card, name="safety_process_card"),
    # 5S Pareto
    path("pareto/", views.five_s_pareto, name="safety_pareto"),
    # Dashboard
    path("dashboard/", views.safety_dashboard, name="safety_dashboard"),
]
