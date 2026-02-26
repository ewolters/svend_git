"""ISO 9001 QMS API URLs."""

from django.urls import path
from . import iso_views

urlpatterns = [
    # Dashboard
    path("dashboard/", iso_views.iso_dashboard, name="iso_dashboard"),

    # NCR Tracker (clause 10.2)
    path("ncrs/", iso_views.ncr_list_create, name="iso_ncr_list"),
    path("ncrs/stats/", iso_views.ncr_stats, name="iso_ncr_stats"),
    path("ncrs/<uuid:ncr_id>/", iso_views.ncr_detail, name="iso_ncr_detail"),
    path("ncrs/<uuid:ncr_id>/launch-rca/", iso_views.ncr_launch_rca, name="iso_ncr_launch_rca"),
    path("ncrs/<uuid:ncr_id>/files/", iso_views.ncr_files, name="iso_ncr_files"),

    # Internal Audits (clause 9.2)
    path("audits/", iso_views.audit_list_create, name="iso_audit_list"),
    path("audits/<uuid:audit_id>/", iso_views.audit_detail, name="iso_audit_detail"),
    path("audits/<uuid:audit_id>/findings/", iso_views.audit_finding_create, name="iso_audit_finding"),

    # Training Matrix (clause 7.2)
    path("training/", iso_views.training_list_create, name="iso_training_list"),
    path("training/<uuid:req_id>/", iso_views.training_detail, name="iso_training_detail"),
    path("training/<uuid:req_id>/records/", iso_views.training_record_create, name="iso_training_record"),
    path("training/records/<uuid:record_id>/", iso_views.training_record_update, name="iso_training_record_update"),

    # Management Reviews (clause 9.3)
    path("reviews/", iso_views.review_list_create, name="iso_review_list"),
    path("reviews/<uuid:review_id>/", iso_views.review_detail, name="iso_review_detail"),

    # Document Control (clause 7.5) — skeleton
    path("documents/", iso_views.document_list_create, name="iso_document_list"),
    path("documents/<uuid:doc_id>/", iso_views.document_detail, name="iso_document_detail"),

    # Supplier Management (clause 8.4) — skeleton
    path("suppliers/", iso_views.supplier_list_create, name="iso_supplier_list"),
    path("suppliers/<uuid:supplier_id>/", iso_views.supplier_detail, name="iso_supplier_detail"),
]
