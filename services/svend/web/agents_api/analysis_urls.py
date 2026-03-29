"""URL routing for analysis/ dispatch — independent of dsw/.

CR: b3b2bf39
"""

from django.urls import path

from .analysis.dispatch import run_analysis
from .analysis.excel_export import export_xlsx, export_xlsx_inline

urlpatterns = [
    path("run/", run_analysis, name="analysis_run"),
    path("export/xlsx/<str:result_id>/", export_xlsx, name="analysis_export_xlsx"),
    path("export/xlsx/", export_xlsx_inline, name="analysis_export_xlsx_inline"),
]
