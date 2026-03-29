"""URL configuration for Svend."""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.contrib.auth.decorators import login_required
from django.contrib.sitemaps import Sitemap
from django.contrib.sitemaps.views import sitemap
from django.urls import include, path
from django.views.generic import RedirectView, TemplateView


def _app_view(template_name, **kwargs):
    """TemplateView wrapped with login_required for /app/* routes."""
    return login_required(TemplateView.as_view(template_name=template_name, **kwargs))


from agents_api.whiteboard_views import guest_board_view
from api.blog_views import blog_detail, blog_list
from api.internal_views import dashboard_view
from api.landing_views import (
    ci_hub_view,
    education_view,
    five_s_playbook_view,
    hoshin_playbook_view,
    iso_audit_playbook_view,
    iso_qms_view,
    kaizen_playbook_view,
    landing_view,
    lsw_playbook_view,
    mdi_playbook_view,
    partnerships_view,
    register_view,
    roadmap_view,
    svend_vs_jmp_view,
    svend_vs_minitab_view,
    vsm_playbook_view,
)
from api.models import BlogPost, WhitePaper
from api.views import compliance_data, compliance_page
from api.whitepaper_views import whitepaper_detail, whitepaper_list, whitepaper_pdf
from loop.views import auditor_portal_view
from syn.varta.urls import urlpatterns as varta_urls

# ---------------------------------------------------------------------------
# ToolRouter (ARCH-001 §10.1) — pluggable QMS tool URL registration
# ---------------------------------------------------------------------------


def _get_tool_router_urls():
    """Import tool registry and return generated URL patterns.

    Deferred to function to avoid circular imports at module load time.
    Each ToolRouter pattern already has the slug prefix (e.g. ishikawa/sessions/),
    so we mount them under api/ to get /api/ishikawa/sessions/.
    """
    from agents_api.tool_registry import register_tools
    from agents_api.tool_router import ToolRouter

    register_tools()
    return ToolRouter.get_urlpatterns()


def _a3_remove_diagram(request, report_id, diagram_id):
    """Thin proxy for A3 remove_diagram — parameterized action can't use ToolRouter."""
    from accounts.permissions import gated_paid
    from agents_api.a3_views import remove_diagram

    return gated_paid(remove_diagram)(request, report_id, diagram_id)


# ---------------------------------------------------------------------------
# Sitemaps
# ---------------------------------------------------------------------------


class StaticSitemap(Sitemap):
    changefreq = "weekly"
    priority = 0.8
    protocol = "https"

    def items(self):
        return [
            "/",
            "/blog/",
            "/privacy/",
            "/terms/",
            "/whitepapers/",
            "/tools/",
            "/tools/cpk-calculator/",
            "/tools/sample-size-calculator/",
            "/tools/oee-calculator/",
            "/tools/sigma-calculator/",
            "/tools/takt-time-calculator/",
            "/tools/kanban-card-generator/",
            "/tools/control-chart-generator/",
            "/tools/gage-rr-calculator/",
            "/tools/pareto-chart-generator/",
            "/tools/bayesian-cpk-calculator/",
            "/tools/fmea-rpn-calculator/",
            "/tools/fpy-rty-calculator/",
            "/svend-vs-minitab/",
            "/svend-vs-jmp/",
            "/classical-vs-bayesian-spc/",
            "/tools/iso-9001-audit-checklist/",
            "/iso-9001-qms-software/",
            "/iso-9001-internal-audit-playbook/",
            "/tools/iso-document-creator/",
            "/for-education/",
            "/continuous-improvement-software/",
            "/managing-for-daily-improvement/",
            "/hoshin-kanri-strategy-deployment/",
            "/kaizen-execution-guide/",
            "/5s-operational-excellence/",
            "/leadership-standard-work/",
            "/value-stream-mapping-methodology/",
            "/partnerships/",
            "/compliance/",
            "/roadmap/",
        ]

    def location(self, item):
        return item


class BlogSitemap(Sitemap):
    changefreq = "weekly"
    priority = 0.7
    protocol = "https"

    def items(self):
        return BlogPost.objects.filter(status=BlogPost.Status.PUBLISHED)

    def lastmod(self, obj):
        return obj.updated_at

    def location(self, obj):
        return f"/blog/{obj.slug}/"


class WhitePaperSitemap(Sitemap):
    changefreq = "monthly"
    priority = 0.8
    protocol = "https"

    def items(self):
        return WhitePaper.objects.filter(status=WhitePaper.Status.PUBLISHED)

    def lastmod(self, obj):
        return obj.updated_at

    def location(self, obj):
        return f"/whitepapers/{obj.slug}/"


sitemaps = {
    "static": StaticSitemap,
    "blog": BlogSitemap,
    "whitepapers": WhitePaperSitemap,
}


# Варта honeypot URLs (must be before real routes to catch scanner paths)
urlpatterns = varta_urls + [
    path("", landing_view, name="home"),
    path("login/", TemplateView.as_view(template_name="login.html"), name="login"),
    path("register/", register_view, name="register"),
    path(
        "verify",
        TemplateView.as_view(template_name="verify_email.html"),
        name="verify_email",
    ),
    path(
        "safety/",
        TemplateView.as_view(template_name="safety_coming_soon.html"),
        name="safety",
    ),
    path("privacy/", TemplateView.as_view(template_name="privacy.html"), name="privacy"),
    path("terms/", TemplateView.as_view(template_name="terms.html"), name="terms"),
    path("app/", _app_view("dashboard.html"), name="app"),
    path("app/dsw/", RedirectView.as_view(url="/app/analysis/", permanent=True), name="dsw"),
    path("app/analysis/", _app_view("analysis_workbench.html"), name="analysis"),
    path("app/workflows/", _app_view("workflows.html"), name="workflows"),
    path("app/triage/", _app_view("triage.html"), name="triage"),
    path("app/forge/", _app_view("forge.html"), name="forge_ui"),
    path("app/whiteboard/", _app_view("whiteboard.html"), name="whiteboard"),
    path(
        "app/whiteboard/<str:room_code>/",
        _app_view("whiteboard.html"),
        name="whiteboard_room",
    ),
    path("app/whiteboard/guest/<str:token>/", guest_board_view, name="whiteboard_guest"),
    path("app/vsm/", _app_view("vsm.html"), name="vsm"),
    path("app/vsm/<uuid:vsm_id>/", _app_view("vsm.html"), name="vsm_edit"),
    path("app/simulator/", _app_view("simulator.html"), name="simulator"),
    path("app/simulator/<uuid:sim_id>/", _app_view("simulator.html"), name="simulator_edit"),
    path("app/onboarding/", _app_view("onboarding.html"), name="onboarding"),
    path("app/settings/", _app_view("settings.html"), name="settings"),
    path("app/models/", _app_view("models.html"), name="models"),
    path("app/forecast/", _app_view("forecast.html"), name="forecast"),
    path("app/projects/", _app_view("projects.html"), name="projects"),
    path("app/problems/", _app_view("projects.html"), name="problems"),  # Legacy redirect
    path("app/hypotheses/", _app_view("projects.html"), name="hypotheses"),  # Legacy redirect
    # path("app/knowledge/", _app_view("knowledge.html"), name="knowledge"),  # Disabled - prototype only
    path("app/experimenter/", _app_view("experimenter.html"), name="experimenter"),
    path("app/spc/", RedirectView.as_view(url="/app/analysis/", permanent=True), name="spc"),
    path("app/safety/", _app_view("safety_app.html"), name="safety_app"),
    # path("app/coder/", _app_view("coder.html"), name="coder"),  # Temporarily disabled
    path("app/a3/", _app_view("a3.html"), name="a3"),
    path("app/a3/<uuid:report_id>/", _app_view("a3.html"), name="a3_edit"),
    path("app/report/", _app_view("report.html"), name="report"),
    path("app/report/<uuid:report_id>/", _app_view("report.html"), name="report_edit"),
    path("app/calculators/", _app_view("calculators.html"), name="calculators"),
    path("app/rca/", _app_view("rca.html"), name="rca"),
    path("app/ishikawa/", _app_view("ishikawa.html"), name="ishikawa"),
    path("app/ishikawa/<uuid:diagram_id>/", _app_view("ishikawa.html"), name="ishikawa_edit"),
    path("app/ce-matrix/", _app_view("ce_matrix.html"), name="ce_matrix"),
    path("app/ce-matrix/<uuid:matrix_id>/", _app_view("ce_matrix.html"), name="ce_matrix_edit"),
    path("app/learn/", _app_view("learn.html"), name="learn"),
    path("app/fmea/", _app_view("fmea.html"), name="fmea"),
    path("app/fmea/<uuid:fmea_id>/", _app_view("fmea.html"), name="fmea_edit"),
    path("app/kanban-cards/", _app_view("kanban_cards.html"), name="kanban_cards"),
    path("app/hoshin/", _app_view("hoshin.html"), name="hoshin"),
    path("app/investigations/", _app_view("investigations.html"), name="investigations"),
    path("app/notebooks/", _app_view("notebooks.html"), name="notebooks"),
    path("app/front-page/", _app_view("front_page.html"), name="front_page"),
    path("app/practice/", _app_view("harada.html"), name="harada"),
    path("app/iso/", _app_view("iso.html"), name="iso"),
    path("app/iso-docs/", _app_view("iso_doc.html"), name="iso_doc"),
    path("app/iso-docs/<uuid:doc_id>/", _app_view("iso_doc.html"), name="iso_doc_edit"),
    # Whitepapers (public, no auth — SEO + PDF download)
    path("whitepapers/", whitepaper_list, name="whitepapers"),
    path("whitepapers/<slug:slug>/", whitepaper_detail, name="whitepaper_detail"),
    path("whitepapers/<slug:slug>/pdf/", whitepaper_pdf, name="whitepaper_pdf"),
    # Free tools (public, no auth — SEO landing pages)
    path(
        "tools/",
        TemplateView.as_view(template_name="tools/index.html"),
        name="tools_index",
    ),
    path(
        "tools/cpk-calculator/",
        TemplateView.as_view(template_name="tools/cpk_calculator.html"),
        name="tool_cpk",
    ),
    path(
        "tools/sample-size-calculator/",
        TemplateView.as_view(template_name="tools/sample_size_calculator.html"),
        name="tool_sample_size",
    ),
    path(
        "tools/oee-calculator/",
        TemplateView.as_view(template_name="tools/oee_calculator.html"),
        name="tool_oee",
    ),
    path(
        "tools/sigma-calculator/",
        TemplateView.as_view(template_name="tools/sigma_calculator.html"),
        name="tool_sigma",
    ),
    path(
        "tools/takt-time-calculator/",
        TemplateView.as_view(template_name="tools/takt_time_calculator.html"),
        name="tool_takt_time",
    ),
    path(
        "tools/kanban-card-generator/",
        TemplateView.as_view(template_name="tools/kanban_card_generator.html"),
        name="tool_kanban",
    ),
    path(
        "tools/control-chart-generator/",
        TemplateView.as_view(template_name="tools/control_chart_generator.html"),
        name="tool_control_chart",
    ),
    path(
        "tools/gage-rr-calculator/",
        TemplateView.as_view(template_name="tools/gage_rr_calculator.html"),
        name="tool_gage_rr",
    ),
    path(
        "tools/pareto-chart-generator/",
        TemplateView.as_view(template_name="tools/pareto_chart_generator.html"),
        name="tool_pareto",
    ),
    path(
        "tools/bayesian-cpk-calculator/",
        TemplateView.as_view(template_name="tools/bayesian_cpk_calculator.html"),
        name="tool_bayesian_cpk",
    ),
    path(
        "tools/fmea-rpn-calculator/",
        TemplateView.as_view(template_name="tools/fmea_rpn_calculator.html"),
        name="tool_fmea_rpn",
    ),
    path(
        "tools/fpy-rty-calculator/",
        TemplateView.as_view(template_name="tools/fpy_rty_calculator.html"),
        name="tool_fpy_rty",
    ),
    path(
        "tools/iso-9001-audit-checklist/",
        TemplateView.as_view(template_name="tools/iso_9001_audit_checklist.html"),
        name="tool_audit_checklist",
    ),
    path(
        "tools/iso-document-creator/",
        TemplateView.as_view(template_name="tools/iso_document_creator.html"),
        name="tool_iso_doc",
    ),
    # Comparison pages (public, no auth — SEO)
    path("svend-vs-minitab/", svend_vs_minitab_view, name="svend_vs_minitab"),
    path("svend-vs-jmp/", svend_vs_jmp_view, name="svend_vs_jmp"),
    path(
        "classical-vs-bayesian-spc/",
        TemplateView.as_view(template_name="classical_vs_bayesian_spc.html"),
        name="classical_vs_bayesian_spc",
    ),
    path("iso-9001-qms-software/", iso_qms_view, name="iso_9001_qms"),
    path(
        "iso-9001-internal-audit-playbook/",
        iso_audit_playbook_view,
        name="iso_audit_playbook",
    ),
    path("partnerships/", partnerships_view, name="partnerships"),
    path("for-education/", education_view, name="education_partnerships"),
    # Continuous Improvement landing pages (public, no auth — SEO)
    path("continuous-improvement-software/", ci_hub_view, name="ci_hub"),
    path("managing-for-daily-improvement/", mdi_playbook_view, name="mdi_playbook"),
    path(
        "hoshin-kanri-strategy-deployment/",
        hoshin_playbook_view,
        name="hoshin_playbook",
    ),
    path("kaizen-execution-guide/", kaizen_playbook_view, name="kaizen_playbook"),
    path("5s-operational-excellence/", five_s_playbook_view, name="five_s_playbook"),
    path("leadership-standard-work/", lsw_playbook_view, name="lsw_playbook"),
    path("value-stream-mapping-methodology/", vsm_playbook_view, name="vsm_playbook"),
    # Public roadmap
    path("roadmap/", roadmap_view, name="roadmap"),
    # Compliance (public, no auth — trust signal for prospects)
    path("compliance/", compliance_page, name="compliance"),
    path("compliance/data/", compliance_data, name="compliance_data"),
    # Blog (public, no auth)
    path("blog/", blog_list, name="blog_list"),
    path("blog/<slug:slug>/", blog_detail, name="blog_detail"),
    # SEO
    path(
        "robots.txt",
        TemplateView.as_view(template_name="robots.txt", content_type="text/plain"),
    ),
    path(
        "sitemap.xml",
        sitemap,
        {"sitemaps": sitemaps},
        name="django.contrib.sitemaps.views.sitemap",
    ),
    path("internal/dashboard/", dashboard_view, name="internal-dashboard"),
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),
    path("api/files/", include("files.urls")),
    path("api/forge/", include("forge.urls")),
    path("api/agents/", include("agents_api.urls")),
    path("api/workflows/", include("agents_api.workflow_urls")),
    path("api/dsw/", include("agents_api.dsw_urls")),
    path("api/analysis/", include("agents_api.analysis_urls")),
    path("api/triage/", include("agents_api.triage_urls")),
    path("api/forecast/", include("agents_api.forecast_urls")),
    path("api/experimenter/", include("agents_api.experimenter_urls")),
    path("api/spc/", include("agents_api.spc_urls")),
    path("api/synara/", include("agents_api.synara_urls")),
    path("api/whiteboard/", include("agents_api.whiteboard_urls")),  # Collaborative whiteboards
    path("api/guide/", include("agents_api.guide_urls")),  # AI guide (rate-limited)
    path("api/reports/", include("agents_api.report_urls")),  # CAPA, 8D reports
    path("api/plantsim/", include("agents_api.plantsim_urls")),  # Plant Simulator (DES)
    path("api/learn/", include("agents_api.learn_urls")),  # Learning module & certification
    path("api/", include(_get_tool_router_urls())),  # ToolRouter: Ishikawa, C&E, A3, VSM, RCA (ARCH-001 §10.1)
    # A3 remove_diagram — parameterized action, not expressible in ToolRouter
    path("api/a3/<uuid:report_id>/diagram/<str:diagram_id>/", _a3_remove_diagram, name="a3-remove-diagram"),
    path("api/fmea/", include("agents_api.fmea_urls")),  # FMEA with Bayesian evidence linking
    path("api/hoshin/", include("agents_api.hoshin_urls")),  # Hoshin Kanri CI (Enterprise)
    path("api/qms/", include("agents_api.qms_urls")),  # QMS cross-module dashboard (Phase 3)
    path("api/iso/", include("agents_api.iso_urls")),  # ISO 9001 QMS (Team/Enterprise)
    path("api/notifications/", include("notifications.urls")),  # NTF-001
    path("api/loop/", include("loop.urls")),  # LOOP-001: Signals, Commitments, Transitions
    path(
        "app/loop/",
        TemplateView.as_view(template_name="loop_dashboard.html"),
        name="loop_dashboard",
    ),  # LOOP-001 §16.2: Accountability Dashboard
    path(
        "app/loop/investigations/<uuid:investigation_id>/",
        TemplateView.as_view(template_name="loop_investigation.html"),
        name="loop_investigation_workspace",
    ),  # LOOP-001 §16.3: Investigation Workspace
    path(
        "app/loop/fmis/",
        TemplateView.as_view(template_name="loop_fmis.html"),
        name="loop_fmis_view",
    ),  # LOOP-001 §16.5: FMIS Global Risk Landscape
    path(
        "app/loop/pc/<uuid:pc_id>/",
        TemplateView.as_view(template_name="loop_pc.html"),
        name="loop_pc_view",
    ),  # LOOP-001 §16.4: Process Confirmation (mobile)
    path(
        "app/loop/pc/new/",
        TemplateView.as_view(template_name="loop_pc.html"),
        name="loop_pc_new",
    ),  # New PC
    # ── Loop Shell (Object 271 QMS redesign) ──
    path(
        "app/loop/detect/signals/",
        TemplateView.as_view(
            template_name="loop_detect_signals.html",
            extra_context={"loop_section": "signals"},
        ),
        name="loop_detect_signals",
    ),
    path(
        "app/loop/detect/conditions/",
        TemplateView.as_view(
            template_name="loop_detect_signals.html",
            extra_context={"loop_section": "conditions"},
        ),
        name="loop_detect_conditions",
    ),
    path(
        "app/loop/detect/fmis/",
        TemplateView.as_view(
            template_name="loop_fmis.html",
            extra_context={"loop_section": "fmis"},
        ),
        name="loop_detect_fmis",
    ),
    path(
        "app/loop/investigate/",
        TemplateView.as_view(
            template_name="loop_section.html",
            extra_context={
                "loop_section": "investigate_active",
                "section_title": "Investigations",
                "filters": True,
                "section_config": '{"section_title":"Investigations","api_endpoint":"/api/loop/dashboard/","response_key":"investigations","title_field":"title","status_field":"status","time_field":"created_at","filter_field":"status","default_filter":"all","filters":[{"label":"Active","value":"active"},{"label":"Concluded","value":"concluded"},{"label":"All","value":"all"}],"detail_fields":[{"key":"description","label":"Description"},{"key":"status","label":"Status"},{"key":"created_at","label":"Created","type":"datetime"}]}',
            },
        ),
        name="loop_investigate",
    ),
    path(
        "app/loop/standardize/commitments/",
        TemplateView.as_view(
            template_name="loop_commitments.html",
            extra_context={"loop_section": "commitments"},
        ),
        name="loop_standardize_commitments",
    ),
    path(
        "app/loop/standardize/policies/",
        TemplateView.as_view(
            template_name="loop_policy.html",
            extra_context={"loop_section": "policies"},
        ),
        name="loop_standardize_policies",
    ),
    path(
        "app/loop/verify/pc/",
        TemplateView.as_view(
            template_name="loop_section.html",
            extra_context={
                "loop_section": "pc",
                "section_title": "Process Confirmations",
                "filters": True,
                "section_config": '{"section_title":"Process Confirmations","api_endpoint":"/api/loop/pcs/","response_key":"pcs","title_field":"process_area","status_field":"diagnosis","time_field":"created_at","subtitle_field":"observer_notes","filter_field":"diagnosis","default_filter":"all","filters":[{"label":"All","value":"all"},{"label":"System Works","value":"system_works"},{"label":"Process Gap","value":"process_gap"},{"label":"Standard Unclear","value":"standard_unclear"}],"detail_fields":[{"key":"process_area","label":"Process Area"},{"key":"diagnosis","label":"Diagnosis"},{"key":"observer_notes","label":"Observer Notes"},{"key":"pass_rate","label":"Pass Rate"},{"key":"created_at","label":"Date","type":"datetime"}]}',
            },
        ),
        name="loop_verify_pc",
    ),
    path(
        "app/loop/verify/fft/",
        TemplateView.as_view(
            template_name="loop_section.html",
            extra_context={
                "loop_section": "fft",
                "section_title": "Forced Failure Tests",
                "filters": True,
                "section_config": '{"section_title":"Forced Failure Tests","api_endpoint":"/api/loop/ffts/","response_key":"ffts","title_field":"control_being_tested","status_field":"result","time_field":"conducted_at","filter_field":"result","default_filter":"all","filters":[{"label":"All","value":"all"},{"label":"Detected","value":"detected"},{"label":"Not Detected","value":"not_detected"},{"label":"Partial","value":"partially_detected"}],"detail_fields":[{"key":"control_being_tested","label":"Control Tested"},{"key":"test_plan","label":"Test Plan"},{"key":"result","label":"Result"},{"key":"detection_count","label":"Detected"},{"key":"injection_count","label":"Injected"},{"key":"conducted_at","label":"Conducted","type":"datetime"}]}',
            },
        ),
        name="loop_verify_fft",
    ),
    # Detect — additional
    path(
        "app/loop/detect/complaints/",
        TemplateView.as_view(
            template_name="loop_section.html",
            extra_context={
                "loop_section": "complaints",
                "section_title": "Customer Complaints",
                "filters": True,
                "section_config": '{"section_title":"Complaints","api_endpoint":"/api/iso/complaints/","title_field":"title","status_field":"status","time_field":"date_received","subtitle_field":"customer_name","filter_field":"status","default_filter":"all","filters":[{"label":"All","value":"all"},{"label":"Open","value":"open"},{"label":"Investigating","value":"investigating"},{"label":"Resolved","value":"resolved"}],"detail_fields":[{"key":"description","label":"Description"},{"key":"customer_name","label":"Customer"},{"key":"product_service","label":"Product/Service"},{"key":"severity","label":"Severity"},{"key":"root_cause","label":"Root Cause"},{"key":"resolution","label":"Resolution"},{"key":"date_received","label":"Received","type":"date"}]}',
            },
        ),
        name="loop_detect_complaints",
    ),
    path(
        "app/loop/detect/supplier/",
        TemplateView.as_view(
            template_name="loop_supplier.html",
            extra_context={"loop_section": "supplier"},
        ),
        name="loop_detect_supplier",
    ),
    # Investigate — concluded
    path(
        "app/loop/investigate/concluded/",
        TemplateView.as_view(
            template_name="loop_section.html",
            extra_context={
                "loop_section": "investigate_concluded",
                "section_title": "Concluded Investigations",
                "filters": True,
                "section_config": '{"section_title":"Concluded","api_endpoint":"/api/loop/dashboard/","response_key":"investigations","title_field":"title","status_field":"status","time_field":"created_at","filter_field":"status","default_filter":"concluded","filters":[{"label":"Concluded","value":"concluded"},{"label":"Exported","value":"exported"},{"label":"All","value":"all"}],"detail_fields":[{"key":"description","label":"Description"},{"key":"status","label":"Status"},{"key":"created_at","label":"Created","type":"datetime"}]}',
            },
        ),
        name="loop_investigate_concluded",
    ),
    # Standardize — additional
    path(
        "app/loop/standardize/documents/",
        TemplateView.as_view(
            template_name="loop_section.html",
            extra_context={
                "loop_section": "documents",
                "section_title": "Controlled Documents",
                "filters": True,
                "section_config": '{"section_title":"Documents","api_endpoint":"/api/iso/documents/","title_field":"title","status_field":"status","time_field":"updated_at","subtitle_field":"document_number","filter_field":"status","default_filter":"all","filters":[{"label":"All","value":"all"},{"label":"Active","value":"active"},{"label":"Draft","value":"draft"},{"label":"Under Review","value":"under_review"}],"detail_fields":[{"key":"title","label":"Title"},{"key":"document_number","label":"Document Number"},{"key":"version","label":"Version"},{"key":"status","label":"Status"},{"key":"iso_clause","label":"ISO Clause"},{"key":"updated_at","label":"Last Updated","type":"datetime"}]}',
            },
        ),
        name="loop_standardize_documents",
    ),
    path(
        "app/loop/standardize/training/",
        TemplateView.as_view(
            template_name="loop_section.html",
            extra_context={
                "loop_section": "training",
                "section_title": "Training",
                "filters": True,
                "section_config": '{"section_title":"Training","api_endpoint":"/api/iso/training/","title_field":"title","status_field":"status","time_field":"due_date","subtitle_field":"employee_name","filter_field":"status","default_filter":"all","filters":[{"label":"All","value":"all"},{"label":"Due","value":"due"},{"label":"Complete","value":"complete"},{"label":"Expired","value":"expired"}],"detail_fields":[{"key":"title","label":"Training"},{"key":"employee_name","label":"Employee"},{"key":"status","label":"Status"},{"key":"due_date","label":"Due Date","type":"date"},{"key":"completed_at","label":"Completed","type":"date"}]}',
            },
        ),
        name="loop_standardize_training",
    ),
    path(
        "app/loop/standardize/ncr/",
        TemplateView.as_view(
            template_name="loop_section.html",
            extra_context={
                "loop_section": "ncr",
                "section_title": "NCR Tracker",
                "filters": True,
                "section_config": '{"section_title":"NCRs","api_endpoint":"/api/iso/ncrs/","title_field":"title","status_field":"status","time_field":"created_at","subtitle_field":"source","filter_field":"status","default_filter":"all","filters":[{"label":"All","value":"all"},{"label":"Open","value":"open"},{"label":"Investigating","value":"investigating"},{"label":"Closed","value":"closed"}],"detail_fields":[{"key":"title","label":"Title"},{"key":"description","label":"Description"},{"key":"source","label":"Source"},{"key":"severity","label":"Severity"},{"key":"status","label":"Status"},{"key":"root_cause","label":"Root Cause"},{"key":"corrective_action","label":"Corrective Action"},{"key":"created_at","label":"Created","type":"datetime"}]}',
            },
        ),
        name="loop_standardize_ncr",
    ),
    path(
        "app/loop/standardize/capa/",
        TemplateView.as_view(
            template_name="loop_section.html",
            extra_context={
                "loop_section": "capa",
                "section_title": "CAPA / 8D",
                "filters": True,
                "section_config": '{"section_title":"CAPA","api_endpoint":"/api/iso/capas/","title_field":"title","status_field":"status","time_field":"created_at","subtitle_field":"source_type","filter_field":"status","default_filter":"all","filters":[{"label":"All","value":"all"},{"label":"Open","value":"open"},{"label":"Investigation","value":"investigation"},{"label":"Verification","value":"verification"},{"label":"Closed","value":"closed"}],"detail_fields":[{"key":"title","label":"Title"},{"key":"description","label":"Description"},{"key":"source_type","label":"Source"},{"key":"priority","label":"Priority"},{"key":"root_cause","label":"Root Cause"},{"key":"corrective_action","label":"Corrective Action"},{"key":"preventive_action","label":"Preventive Action"},{"key":"created_at","label":"Created","type":"datetime"}]}',
            },
        ),
        name="loop_standardize_capa",
    ),
    # Verify — additional
    path(
        "app/loop/verify/audits/",
        TemplateView.as_view(
            template_name="loop_section.html",
            extra_context={
                "loop_section": "audits",
                "section_title": "Internal Audits",
                "filters": True,
                "section_config": '{"section_title":"Audits","api_endpoint":"/api/iso/audits/","title_field":"title","status_field":"status","time_field":"scheduled_date","filter_field":"status","default_filter":"all","filters":[{"label":"All","value":"all"},{"label":"Scheduled","value":"scheduled"},{"label":"In Progress","value":"in_progress"},{"label":"Completed","value":"completed"}],"detail_fields":[{"key":"title","label":"Title"},{"key":"scope","label":"Scope"},{"key":"status","label":"Status"},{"key":"scheduled_date","label":"Scheduled","type":"date"},{"key":"findings_count","label":"Findings"}]}',
            },
        ),
        name="loop_verify_audits",
    ),
    path(
        "app/loop/verify/reviews/",
        TemplateView.as_view(
            template_name="loop_section.html",
            extra_context={
                "loop_section": "reviews",
                "section_title": "Management Reviews",
                "filters": True,
                "section_config": '{"section_title":"Reviews","api_endpoint":"/api/iso/reviews/","title_field":"title","status_field":"status","time_field":"scheduled_date","filter_field":"status","default_filter":"all","filters":[{"label":"All","value":"all"},{"label":"Scheduled","value":"scheduled"},{"label":"Completed","value":"completed"}],"detail_fields":[{"key":"title","label":"Title"},{"key":"status","label":"Status"},{"key":"scheduled_date","label":"Scheduled","type":"date"},{"key":"attendees","label":"Attendees"},{"key":"decisions","label":"Decisions"}]}',
            },
        ),
        name="loop_verify_reviews",
    ),
    # Legacy route
    path(
        "app/loop/policies/",
        TemplateView.as_view(template_name="loop_policy.html"),
        name="loop_policy_config",
    ),  # LOOP-001 §4: QMS Policy Configuration
    path(
        "app/loop/auditor/",
        TemplateView.as_view(template_name="loop_auditor_manage.html"),
        name="loop_auditor_manage",
    ),  # LOOP-001 §11: Auditor Token Management
    path(
        "audit/<str:token>/",
        auditor_portal_view,
        name="loop_auditor_portal",
    ),  # LOOP-001 §11: Auditor Portal (no auth — token in URL)
    path(
        "app/process-map/",
        TemplateView.as_view(template_name="graph_map.html"),
        name="graph_map",
    ),  # GRAPH-001 §15: Process Map (Beta)
    path("api/graph/", include("graph.urls")),  # GRAPH-001 §11: Graph API
    path("api/safety/", include("safety.urls")),  # HIRARC Safety (Enterprise)
    path("api/privacy/", include("accounts.privacy_urls")),  # PRIV-001 (SOC 2 P1.8)
    path("api/capa/", include("agents_api.capa_urls")),  # CAPA standalone (ISO 10.2, FEAT-004)
    path("api/iso-docs/", include("agents_api.iso_doc_urls")),  # ISO Document Creator
    path("api/actions/", include("agents_api.action_urls")),  # Shared action item update/delete
    path(
        "api/investigations/", include("agents_api.investigation_urls")
    ),  # Investigation lifecycle (CANON-002) — deprecated
    path("api/notebooks/", include("agents_api.notebook_urls")),  # Notebook lifecycle (NB-001)
    path("api/harada/", include("agents_api.harada_urls")),  # Harada Method (questionnaire, goals, routines, diary)
    path("api/core/", include("core.urls")),  # Projects, hypotheses, evidence, knowledge graph
    path("api/workbench/", include("workbench.urls")),
    path("chat/", include("chat.urls")),
    path("action/<str:token>/", include("agents_api.token_urls")),  # ActionToken (QMS-002, no auth)
    path("ntf/<str:token>/", include("notifications.token_urls")),  # NotificationToken (NTF-001 §5.2, no auth)
    path("", include("accounts.urls")),  # Billing endpoints
    # Password reset
    path(
        "accounts/password_reset/",
        auth_views.PasswordResetView.as_view(
            template_name="registration/password_reset_form.html",
            email_template_name="registration/password_reset_email.html",
            subject_template_name="registration/password_reset_subject.txt",
        ),
        name="password_reset",
    ),
    path(
        "accounts/password_reset/done/",
        auth_views.PasswordResetDoneView.as_view(
            template_name="registration/password_reset_done.html",
        ),
        name="password_reset_done",
    ),
    path(
        "accounts/reset/<uidb64>/<token>/",
        auth_views.PasswordResetConfirmView.as_view(
            template_name="registration/password_reset_confirm.html",
        ),
        name="password_reset_confirm",
    ),
    path(
        "accounts/reset/done/",
        auth_views.PasswordResetCompleteView.as_view(
            template_name="registration/password_reset_complete.html",
        ),
        name="password_reset_complete",
    ),
]

# Serve media files (models, uploads)
# In production, consider using nginx/whitenoise for static/media
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
