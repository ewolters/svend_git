"""Hoshin Kanri CI project management API URLs.

Enterprise-only — all views gated by @require_feature("hoshin_kanri").
"""

from django.urls import path
from . import hoshin_views as views

urlpatterns = [
    # Sites
    path("sites/", views.list_sites, name="hoshin_sites"),
    path("sites/create/", views.create_site, name="hoshin_create_site"),
    path("sites/<uuid:site_id>/", views.get_site, name="hoshin_get_site"),
    path("sites/<uuid:site_id>/update/", views.update_site, name="hoshin_update_site"),
    path("sites/<uuid:site_id>/delete/", views.delete_site, name="hoshin_delete_site"),

    # Hoshin Projects
    path("projects/", views.list_hoshin_projects, name="hoshin_projects"),
    path("projects/create/", views.create_hoshin_project, name="hoshin_create_project"),
    path("projects/from-proposals/", views.create_from_proposals, name="hoshin_from_proposals"),
    path("projects/<uuid:hoshin_id>/", views.get_hoshin_project, name="hoshin_get_project"),
    path("projects/<uuid:hoshin_id>/update/", views.update_hoshin_project, name="hoshin_update_project"),
    path("projects/<uuid:hoshin_id>/delete/", views.delete_hoshin_project, name="hoshin_delete_project"),
    path("projects/<uuid:hoshin_id>/monthly/<int:month>/", views.update_monthly_actual, name="hoshin_monthly"),

    # Action Items
    path("projects/<uuid:hoshin_id>/actions/", views.list_action_items, name="hoshin_actions"),
    path("projects/<uuid:hoshin_id>/actions/create/", views.create_action_item, name="hoshin_create_action"),
    path("actions/<uuid:action_id>/update/", views.update_action_item, name="hoshin_update_action"),
    path("actions/<uuid:action_id>/delete/", views.delete_action_item, name="hoshin_delete_action"),

    # Dashboard & reference
    path("dashboard/", views.hoshin_dashboard, name="hoshin_dashboard"),
    path("calculation-methods/", views.list_calculation_methods, name="hoshin_calc_methods"),
    path("test-formula/", views.test_formula, name="hoshin_test_formula"),
]
