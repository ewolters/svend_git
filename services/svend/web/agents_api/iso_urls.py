"""ISO 9001 QMS API URLs."""

from django.urls import path

from . import iso_views

urlpatterns = [
    # Dashboard
    path("dashboard/", iso_views.iso_dashboard, name="iso_dashboard"),
    # Team members (for assignment dropdowns)
    path("team-members/", iso_views.team_members, name="iso_team_members"),
    # NCR Tracker (clause 10.2)
    path("ncrs/", iso_views.ncr_list_create, name="iso_ncr_list"),
    path("ncrs/stats/", iso_views.ncr_stats, name="iso_ncr_stats"),
    path("ncrs/analytics/", iso_views.ncr_analytics, name="iso_ncr_analytics"),
    path("ncrs/<uuid:ncr_id>/", iso_views.ncr_detail, name="iso_ncr_detail"),
    path("ncrs/<uuid:ncr_id>/launch-rca/", iso_views.ncr_launch_rca, name="iso_ncr_launch_rca"),
    path("ncrs/<uuid:ncr_id>/files/", iso_views.ncr_files, name="iso_ncr_files"),
    # Internal Audits (clause 9.2)
    path("audits/", iso_views.audit_list_create, name="iso_audit_list"),
    path("audits/<uuid:audit_id>/", iso_views.audit_detail, name="iso_audit_detail"),
    path("audits/<uuid:audit_id>/findings/", iso_views.audit_finding_create, name="iso_audit_finding"),
    # Audit Checklists
    path("checklists/", iso_views.audit_checklist_list_create, name="iso_checklist_list"),
    path("checklists/<uuid:checklist_id>/", iso_views.audit_checklist_detail, name="iso_checklist_detail"),
    # Training Matrix (clause 7.2)
    path("training/", iso_views.training_list_create, name="iso_training_list"),
    path("training/<uuid:req_id>/", iso_views.training_detail, name="iso_training_detail"),
    path("training/<uuid:req_id>/records/", iso_views.training_record_create, name="iso_training_record"),
    path("training/records/<uuid:record_id>/", iso_views.training_record_update, name="iso_training_record_update"),
    path("training/records/<uuid:record_id>/files/", iso_views.training_record_files, name="iso_training_record_files"),
    # Management Reviews (clause 9.3)
    path("reviews/", iso_views.review_list_create, name="iso_review_list"),
    path("reviews/<uuid:review_id>/", iso_views.review_detail, name="iso_review_detail"),
    # Management Review Templates
    path("review-templates/", iso_views.review_template_list_create, name="iso_review_template_list"),
    path("review-templates/default/", iso_views.review_template_default, name="iso_review_template_default"),
    path("review-templates/<uuid:template_id>/", iso_views.review_template_detail, name="iso_review_template_detail"),
    # Document Control (clause 7.5) — skeleton
    path("documents/", iso_views.document_list_create, name="iso_document_list"),
    path("documents/<uuid:doc_id>/", iso_views.document_detail, name="iso_document_detail"),
    path("documents/<uuid:doc_id>/files/", iso_views.document_files, name="iso_document_files"),
    # Supplier Management (clause 8.4) — skeleton
    path("suppliers/", iso_views.supplier_list_create, name="iso_supplier_list"),
    path("suppliers/<uuid:supplier_id>/", iso_views.supplier_detail, name="iso_supplier_detail"),
    # Electronic Signatures (21 CFR Part 11)
    path("signatures/", iso_views.signature_list_or_sign, name="iso_signatures"),
    path("signatures/<uuid:sig_id>/verify/", iso_views.signature_verify, name="iso_signature_verify"),
    path("signatures/verify-chain/", iso_views.signature_verify_chain, name="iso_signature_verify_chain"),
    # Study Output Actions (Phase 7) — QMS routing from Studies
    path("study-actions/raise-capa/", iso_views.study_raise_capa, name="iso_study_raise_capa"),
    path("study-actions/schedule-audit/", iso_views.study_schedule_audit, name="iso_study_schedule_audit"),
    path("study-actions/request-doc-update/", iso_views.study_request_doc_update, name="iso_study_request_doc_update"),
    path("study-actions/flag-training-gap/", iso_views.study_flag_training_gap, name="iso_study_flag_training_gap"),
    path("study-actions/flag-fmea-update/", iso_views.study_flag_fmea_update, name="iso_study_flag_fmea_update"),
]
