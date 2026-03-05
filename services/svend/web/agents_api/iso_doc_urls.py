"""ISO Document Creator API URLs."""

from django.urls import path

from . import iso_doc_views

urlpatterns = [
    # Document types registry
    path("types/", iso_doc_views.list_types, name="iso_doc_types"),
    # Document CRUD
    path("", iso_doc_views.document_list_create, name="iso_doc_list"),
    path("<uuid:doc_id>/", iso_doc_views.document_detail, name="iso_doc_detail"),
    # Section CRUD
    path("<uuid:doc_id>/sections/", iso_doc_views.section_create, name="iso_doc_section_create"),
    path("<uuid:doc_id>/sections/reorder/", iso_doc_views.section_reorder, name="iso_doc_section_reorder"),
    path("<uuid:doc_id>/sections/<uuid:sec_id>/", iso_doc_views.section_detail, name="iso_doc_section_detail"),
    # Media
    path(
        "<uuid:doc_id>/sections/<uuid:sec_id>/embed-whiteboard/",
        iso_doc_views.embed_whiteboard,
        name="iso_doc_embed_wb",
    ),
    # Export
    path("<uuid:doc_id>/export/pdf/", iso_doc_views.export_pdf, name="iso_doc_export_pdf"),
    path("<uuid:doc_id>/export/docx/", iso_doc_views.export_docx, name="iso_doc_export_docx"),
    # Publish to Document Control
    path("<uuid:doc_id>/publish/", iso_doc_views.publish_to_doc_control, name="iso_doc_publish"),
]
