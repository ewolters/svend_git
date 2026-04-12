"""URL routing for /api/analysis/ — forge-only endpoints for the new workbench.

CR: b3b2bf39, 9999588b
"""

from django.urls import path

from . import analysis_views
from .analysis.excel_export import export_xlsx, export_xlsx_inline

urlpatterns = [
    # Forge-only endpoints (new workbench)
    path("run/", analysis_views.run_analysis, name="analysis_run"),
    path("upload-data/", analysis_views.upload_data, name="analysis_upload_data"),
    # Export (shared)
    path("export/xlsx/<str:result_id>/", export_xlsx, name="analysis_export_xlsx"),
    path("export/xlsx/", export_xlsx_inline, name="analysis_export_xlsx_inline"),
]
