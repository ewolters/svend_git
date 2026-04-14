"""Hoshin Kanri CI project management API URLs.

Enterprise-only — all views gated by @require_feature("hoshin_kanri").
"""

from django.urls import path

from . import views
from . import xmatrix_views as views_xm

urlpatterns = [
    # Sites
    path("sites/", views.list_sites, name="hoshin_sites"),
    path("sites/create/", views.create_site, name="hoshin_create_site"),
    path("sites/<uuid:site_id>/", views.get_site, name="hoshin_get_site"),
    path("sites/<uuid:site_id>/update/", views.update_site, name="hoshin_update_site"),
    path("sites/<uuid:site_id>/delete/", views.delete_site, name="hoshin_delete_site"),
    path(
        "sites/<uuid:site_id>/members/",
        views.list_site_members,
        name="hoshin_site_members",
    ),
    path(
        "sites/<uuid:site_id>/members/grant/",
        views.grant_site_access,
        name="hoshin_grant_site_access",
    ),
    path(
        "sites/<uuid:site_id>/members/<uuid:access_id>/revoke/",
        views.revoke_site_access,
        name="hoshin_revoke_site_access",
    ),
    # Hoshin Projects
    path("projects/", views.list_hoshin_projects, name="hoshin_projects"),
    path("projects/create/", views.create_hoshin_project, name="hoshin_create_project"),
    path(
        "projects/from-proposals/",
        views.create_from_proposals,
        name="hoshin_from_proposals",
    ),
    path(
        "projects/<uuid:hoshin_id>/",
        views.get_hoshin_project,
        name="hoshin_get_project",
    ),
    path(
        "projects/<uuid:hoshin_id>/update/",
        views.update_hoshin_project,
        name="hoshin_update_project",
    ),
    path(
        "projects/<uuid:hoshin_id>/delete/",
        views.delete_hoshin_project,
        name="hoshin_delete_project",
    ),
    path(
        "projects/<uuid:hoshin_id>/monthly/<int:month>/",
        views.update_monthly_actual,
        name="hoshin_monthly",
    ),
    # Action Items
    path(
        "projects/<uuid:hoshin_id>/actions/",
        views.list_action_items,
        name="hoshin_actions",
    ),
    path(
        "projects/<uuid:hoshin_id>/actions/create/",
        views.create_action_item,
        name="hoshin_create_action",
    ),
    path(
        "actions/<uuid:action_id>/update/",
        views.update_action_item,
        name="hoshin_update_action",
    ),
    path(
        "actions/<uuid:action_id>/delete/",
        views.delete_action_item,
        name="hoshin_delete_action",
    ),
    # Dashboard, calendar & reference
    path("dashboard/", views.hoshin_dashboard, name="hoshin_dashboard"),
    path("calendar/", views.hoshin_calendar_view, name="hoshin_calendar"),
    path(
        "calendar/facilitators/",
        views.hoshin_calendar_facilitators,
        name="hoshin_calendar_facilitators",
    ),
    path(
        "calculation-methods/",
        views.list_calculation_methods,
        name="hoshin_calc_methods",
    ),
    path("test-formula/", views.test_formula, name="hoshin_test_formula"),
    # X-Matrix
    path("x-matrix/", views_xm.get_xmatrix_data, name="xmatrix_data"),
    path(
        "x-matrix/correlations/",
        views_xm.update_correlation,
        name="xmatrix_correlation",
    ),
    path("x-matrix/rollover/", views_xm.rollover_fiscal_year, name="xmatrix_rollover"),
    # Strategic Objectives
    path(
        "strategic-objectives/",
        views_xm.list_create_strategic_objectives,
        name="xmatrix_strategic",
    ),
    path(
        "strategic-objectives/<uuid:obj_id>/",
        views_xm.update_delete_strategic_objective,
        name="xmatrix_strategic_detail",
    ),
    # Annual Objectives
    path(
        "annual-objectives/",
        views_xm.list_create_annual_objectives,
        name="xmatrix_annual",
    ),
    path(
        "annual-objectives/<uuid:obj_id>/",
        views_xm.update_delete_annual_objective,
        name="xmatrix_annual_detail",
    ),
    # KPIs
    path("kpis/", views_xm.list_create_kpis, name="xmatrix_kpis"),
    path("kpis/<uuid:kpi_id>/", views_xm.update_delete_kpi, name="xmatrix_kpi_detail"),
    # VSM Lifecycle
    path("vsm/<uuid:vsm_id>/promote/", views_xm.promote_vsm, name="xmatrix_vsm_promote"),
    # Intelligence Layer (Phase 3)
    path(
        "sites/<uuid:site_id>/alignment/",
        views.alignment_analysis,
        name="hoshin_alignment",
    ),
    # Employees (QMS-002)
    path("employees/", views.list_create_employees, name="hoshin_employees"),
    path("employees/import/", views.employees_import, name="hoshin_employees_import"),
    path("employees/<uuid:emp_id>/", views.employee_detail, name="hoshin_employee_detail"),
    path(
        "employees/<uuid:emp_id>/availability/",
        views.employee_availability,
        name="hoshin_employee_availability",
    ),
    path(
        "employees/<uuid:emp_id>/timeline/",
        views.employee_timeline,
        name="hoshin_employee_timeline",
    ),
    # Resource Commitments (QMS-002)
    path("commitments/", views.list_create_commitments, name="hoshin_commitments"),
    path(
        "commitments/<uuid:commitment_id>/",
        views.commitment_detail,
        name="hoshin_commitment_detail",
    ),
    # Project Templates
    path("templates/", views.template_list_create, name="hoshin_templates"),
    path("templates/<uuid:tpl_id>/", views.template_detail, name="hoshin_template_detail"),
    path(
        "projects/from-template/",
        views.create_from_template,
        name="hoshin_from_template",
    ),
]
