"""URL configuration for Svend."""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path, include
from django.views.generic import TemplateView

urlpatterns = [
    path("", TemplateView.as_view(template_name="landing.html"), name="home"),
    path("login/", TemplateView.as_view(template_name="login.html"), name="login"),
    path("register/", TemplateView.as_view(template_name="register.html"), name="register"),
    path("verify", TemplateView.as_view(template_name="verify_email.html"), name="verify_email"),
    path("privacy/", TemplateView.as_view(template_name="privacy.html"), name="privacy"),
    path("terms/", TemplateView.as_view(template_name="terms.html"), name="terms"),
    path("app/", TemplateView.as_view(template_name="workbench.html"), name="app"),
    path("app/new/", TemplateView.as_view(template_name="workbench_new.html"), name="app_new"),
    path("app/dsw/", TemplateView.as_view(template_name="dsw.html"), name="dsw"),
    # Legacy redirect: /app/analysis/ serves old analysis workbench
    path("app/analysis/", TemplateView.as_view(template_name="analysis_workbench.html"), name="analysis"),
    path("app/workflows/", TemplateView.as_view(template_name="workflows.html"), name="workflows"),
    path("app/triage/", TemplateView.as_view(template_name="triage.html"), name="triage"),
    path("app/forge/", TemplateView.as_view(template_name="forge.html"), name="forge_ui"),
    path("app/whiteboard/", TemplateView.as_view(template_name="whiteboard.html"), name="whiteboard"),
    path("app/settings/", TemplateView.as_view(template_name="settings.html"), name="settings"),
    path("app/models/", TemplateView.as_view(template_name="models.html"), name="models"),
    path("app/forecast/", TemplateView.as_view(template_name="forecast.html"), name="forecast"),
    path("app/projects/", TemplateView.as_view(template_name="projects.html"), name="projects"),
    path("app/problems/", TemplateView.as_view(template_name="projects.html"), name="problems"),  # Legacy redirect
    path("app/hypotheses/", TemplateView.as_view(template_name="projects.html"), name="hypotheses"),  # Legacy redirect
    path("app/knowledge/", TemplateView.as_view(template_name="knowledge.html"), name="knowledge"),
    path("app/experimenter/", TemplateView.as_view(template_name="experimenter.html"), name="experimenter"),
    path("app/spc/", TemplateView.as_view(template_name="spc.html"), name="spc"),
    # path("app/coder/", TemplateView.as_view(template_name="coder.html"), name="coder"),  # Temporarily disabled
    path("app/learn/", TemplateView.as_view(template_name="learn.html"), name="learn"),
    path("admin/", admin.site.urls),
    path("api/", include("api.urls")),
    path("api/files/", include("files.urls")),
    path("api/forge/", include("forge.urls")),
    path("api/agents/", include("agents_api.urls")),
    path("api/workflows/", include("agents_api.workflow_urls")),
    path("api/dsw/", include("agents_api.dsw_urls")),
    path("api/triage/", include("agents_api.triage_urls")),
    path("api/forecast/", include("agents_api.forecast_urls")),
    path("api/problems/", include("agents_api.problem_urls")),
    path("api/experimenter/", include("agents_api.experimenter_urls")),
    path("api/spc/", include("agents_api.spc_urls")),
    path("api/synara/", include("agents_api.synara_urls")),
    path("api/learn/", include("agents_api.learn_urls")),  # Learning module & certification
    path("api/core/", include("core.urls")),  # Projects, hypotheses, evidence, knowledge graph
    path("api/workbench/", include("workbench.urls")),
    path("chat/", include("chat.urls")),
    path("", include("accounts.urls")),  # Billing endpoints

    # Password reset
    path("accounts/password_reset/", auth_views.PasswordResetView.as_view(
        template_name="registration/password_reset_form.html",
        email_template_name="registration/password_reset_email.html",
        subject_template_name="registration/password_reset_subject.txt",
    ), name="password_reset"),
    path("accounts/password_reset/done/", auth_views.PasswordResetDoneView.as_view(
        template_name="registration/password_reset_done.html",
    ), name="password_reset_done"),
    path("accounts/reset/<uidb64>/<token>/", auth_views.PasswordResetConfirmView.as_view(
        template_name="registration/password_reset_confirm.html",
    ), name="password_reset_confirm"),
    path("accounts/reset/done/", auth_views.PasswordResetCompleteView.as_view(
        template_name="registration/password_reset_complete.html",
    ), name="password_reset_complete"),
]

# Serve media files (models, uploads)
# In production, consider using nginx/whitenoise for static/media
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
