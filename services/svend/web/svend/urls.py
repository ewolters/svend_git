"""URL configuration for Svend."""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.contrib.sitemaps import Sitemap
from django.contrib.sitemaps.views import sitemap
from django.urls import include, path
from django.views.generic import RedirectView, TemplateView

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


urlpatterns = [
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
    path(
        "privacy/", TemplateView.as_view(template_name="privacy.html"), name="privacy"
    ),
    path("terms/", TemplateView.as_view(template_name="terms.html"), name="terms"),
    path("app/", TemplateView.as_view(template_name="dashboard.html"), name="app"),
    path(
        "app/dsw/", TemplateView.as_view(template_name="workbench_new.html"), name="dsw"
    ),
    # Legacy redirect: /app/analysis/ serves old analysis workbench
    path(
        "app/analysis/",
        TemplateView.as_view(template_name="analysis_workbench.html"),
        name="analysis",
    ),
    path(
        "app/workflows/",
        TemplateView.as_view(template_name="workflows.html"),
        name="workflows",
    ),
    path(
        "app/triage/", TemplateView.as_view(template_name="triage.html"), name="triage"
    ),
    path(
        "app/forge/", TemplateView.as_view(template_name="forge.html"), name="forge_ui"
    ),
    path(
        "app/whiteboard/",
        TemplateView.as_view(template_name="whiteboard.html"),
        name="whiteboard",
    ),
    path(
        "app/whiteboard/<str:room_code>/",
        TemplateView.as_view(template_name="whiteboard.html"),
        name="whiteboard_room",
    ),
    path(
        "app/whiteboard/guest/<str:token>/", guest_board_view, name="whiteboard_guest"
    ),
    path("app/vsm/", TemplateView.as_view(template_name="vsm.html"), name="vsm"),
    path(
        "app/vsm/<uuid:vsm_id>/",
        TemplateView.as_view(template_name="vsm.html"),
        name="vsm_edit",
    ),
    path(
        "app/simulator/",
        TemplateView.as_view(template_name="simulator.html"),
        name="simulator",
    ),
    path(
        "app/simulator/<uuid:sim_id>/",
        TemplateView.as_view(template_name="simulator.html"),
        name="simulator_edit",
    ),
    path(
        "app/onboarding/",
        TemplateView.as_view(template_name="onboarding.html"),
        name="onboarding",
    ),
    path(
        "app/settings/",
        TemplateView.as_view(template_name="settings.html"),
        name="settings",
    ),
    path(
        "app/models/", TemplateView.as_view(template_name="models.html"), name="models"
    ),
    path(
        "app/forecast/",
        TemplateView.as_view(template_name="forecast.html"),
        name="forecast",
    ),
    path(
        "app/projects/",
        TemplateView.as_view(template_name="projects.html"),
        name="projects",
    ),
    path(
        "app/problems/",
        TemplateView.as_view(template_name="projects.html"),
        name="problems",
    ),  # Legacy redirect
    path(
        "app/hypotheses/",
        TemplateView.as_view(template_name="projects.html"),
        name="hypotheses",
    ),  # Legacy redirect
    # path("app/knowledge/", TemplateView.as_view(template_name="knowledge.html"), name="knowledge"),  # Disabled - prototype only
    path(
        "app/experimenter/",
        TemplateView.as_view(template_name="experimenter.html"),
        name="experimenter",
    ),
    path("app/spc/", RedirectView.as_view(url="/app/analysis/", permanent=True), name="spc"),
    path(
        "app/safety/",
        TemplateView.as_view(template_name="safety_app.html"),
        name="safety_app",
    ),
    # path("app/coder/", TemplateView.as_view(template_name="coder.html"), name="coder"),  # Temporarily disabled
    path("app/a3/", TemplateView.as_view(template_name="a3.html"), name="a3"),
    path(
        "app/a3/<uuid:report_id>/",
        TemplateView.as_view(template_name="a3.html"),
        name="a3_edit",
    ),
    path(
        "app/report/", TemplateView.as_view(template_name="report.html"), name="report"
    ),
    path(
        "app/report/<uuid:report_id>/",
        TemplateView.as_view(template_name="report.html"),
        name="report_edit",
    ),
    path(
        "app/calculators/",
        TemplateView.as_view(template_name="calculators.html"),
        name="calculators",
    ),
    path("app/rca/", TemplateView.as_view(template_name="rca.html"), name="rca"),
    path(
        "app/ishikawa/",
        TemplateView.as_view(template_name="ishikawa.html"),
        name="ishikawa",
    ),
    path(
        "app/ishikawa/<uuid:diagram_id>/",
        TemplateView.as_view(template_name="ishikawa.html"),
        name="ishikawa_edit",
    ),
    path(
        "app/ce-matrix/",
        TemplateView.as_view(template_name="ce_matrix.html"),
        name="ce_matrix",
    ),
    path(
        "app/ce-matrix/<uuid:matrix_id>/",
        TemplateView.as_view(template_name="ce_matrix.html"),
        name="ce_matrix_edit",
    ),
    path("app/learn/", TemplateView.as_view(template_name="learn.html"), name="learn"),
    path("app/fmea/", TemplateView.as_view(template_name="fmea.html"), name="fmea"),
    path(
        "app/fmea/<uuid:fmea_id>/",
        TemplateView.as_view(template_name="fmea.html"),
        name="fmea_edit",
    ),
    path(
        "app/kanban-cards/",
        TemplateView.as_view(template_name="kanban_cards.html"),
        name="kanban_cards",
    ),
    path(
        "app/hoshin/", TemplateView.as_view(template_name="hoshin.html"), name="hoshin"
    ),
    path(
        "app/investigations/",
        TemplateView.as_view(template_name="investigations.html"),
        name="investigations",
    ),
    path(
        "app/notebooks/",
        TemplateView.as_view(template_name="notebooks.html"),
        name="notebooks",
    ),
    path(
        "app/front-page/",
        TemplateView.as_view(template_name="front_page.html"),
        name="front_page",
    ),
    path(
        "app/practice/",
        TemplateView.as_view(template_name="harada.html"),
        name="harada",
    ),
    path("app/iso/", TemplateView.as_view(template_name="iso.html"), name="iso"),
    path(
        "app/iso-docs/",
        TemplateView.as_view(template_name="iso_doc.html"),
        name="iso_doc",
    ),
    path(
        "app/iso-docs/<uuid:doc_id>/",
        TemplateView.as_view(template_name="iso_doc.html"),
        name="iso_doc_edit",
    ),
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
    path(
        "api/whiteboard/", include("agents_api.whiteboard_urls")
    ),  # Collaborative whiteboards
    path("api/guide/", include("agents_api.guide_urls")),  # AI guide (rate-limited)
    path("api/a3/", include("agents_api.a3_urls")),  # A3 problem-solving reports
    path("api/reports/", include("agents_api.report_urls")),  # CAPA, 8D reports
    path("api/vsm/", include("agents_api.vsm_urls")),  # Value Stream Mapping
    path("api/plantsim/", include("agents_api.plantsim_urls")),  # Plant Simulator (DES)
    path(
        "api/learn/", include("agents_api.learn_urls")
    ),  # Learning module & certification
    path("api/rca/", include("agents_api.rca_urls")),  # Root cause analysis critique
    path(
        "api/ishikawa/", include("agents_api.ishikawa_urls")
    ),  # Ishikawa (fishbone) diagrams
    path("api/ce/", include("agents_api.ce_urls")),  # Cause & Effect matrices
    path(
        "api/fmea/", include("agents_api.fmea_urls")
    ),  # FMEA with Bayesian evidence linking
    path(
        "api/hoshin/", include("agents_api.hoshin_urls")
    ),  # Hoshin Kanri CI (Enterprise)
    path(
        "api/qms/", include("agents_api.qms_urls")
    ),  # QMS cross-module dashboard (Phase 3)
    path("api/iso/", include("agents_api.iso_urls")),  # ISO 9001 QMS (Team/Enterprise)
    path("api/notifications/", include("notifications.urls")),  # NTF-001
    path("api/safety/", include("safety.urls")),  # HIRARC Safety (Enterprise)
    path("api/privacy/", include("accounts.privacy_urls")),  # PRIV-001 (SOC 2 P1.8)
    path(
        "api/capa/", include("agents_api.capa_urls")
    ),  # CAPA standalone (ISO 10.2, FEAT-004)
    path("api/iso-docs/", include("agents_api.iso_doc_urls")),  # ISO Document Creator
    path(
        "api/actions/", include("agents_api.action_urls")
    ),  # Shared action item update/delete
    path(
        "api/investigations/", include("agents_api.investigation_urls")
    ),  # Investigation lifecycle (CANON-002) — deprecated
    path(
        "api/notebooks/", include("agents_api.notebook_urls")
    ),  # Notebook lifecycle (NB-001)
    path(
        "api/harada/", include("agents_api.harada_urls")
    ),  # Harada Method (questionnaire, goals, routines, diary)
    path(
        "api/core/", include("core.urls")
    ),  # Projects, hypotheses, evidence, knowledge graph
    path("api/workbench/", include("workbench.urls")),
    path("chat/", include("chat.urls")),
    path(
        "action/<str:token>/", include("agents_api.token_urls")
    ),  # ActionToken (QMS-002, no auth)
    path(
        "ntf/<str:token>/", include("notifications.token_urls")
    ),  # NotificationToken (NTF-001 §5.2, no auth)
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
