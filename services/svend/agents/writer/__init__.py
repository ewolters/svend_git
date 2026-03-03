"""Writer Agent - Document generation with templates and citations."""
from .agent import (
    WriterAgent,
    DocumentRequest,
    DocumentType,
    GeneratedDocument,
    write_from_research,
    write_with_style,
    quick_write,
    # Quality assessment
    WriterQualityAssessment,
    TemplateComplianceResult,
    assess_document_quality,
    check_template_compliance,
)

__all__ = [
    "WriterAgent",
    "DocumentRequest",
    "DocumentType",
    "GeneratedDocument",
    "write_from_research",
    "write_with_style",
    "quick_write",
    # Quality assessment
    "WriterQualityAssessment",
    "TemplateComplianceResult",
    "assess_document_quality",
    "check_template_compliance",
]
