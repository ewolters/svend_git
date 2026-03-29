# Document Builder Service — S3 Specification

**Date:** 2026-03-29
**Author:** S3 (Infrastructure / Document System)
**Status:** SPEC — awaiting Eric approval before build
**CR:** Will create on approval

---

## What This Is

A unified document rendering service that replaces 6 scattered WeasyPrint call sites and 1 python-docx call site with a single service. Every document the platform produces — PDFs, DOCX, HTML reports — routes through one path with consistent branding, configurable templates, and tenant-aware output.

The service also establishes the foundation for OLR-001 document generation: investigations auto-assemble CAPA-format compliance artifacts, control plans render from graph state, and any structured data in the platform can become a formatted document.

## What This Is NOT

- Not a document editor (that's the ISO doc system in `iso_doc_views.py`)
- Not a template marketplace (future)
- Not a replacement for existing views yet — they continue working. Migration is a separate task.

---

## Architecture

```
DocumentService
├── Template Registry (what documents exist)
│   ├── System templates (CAPA, A3, control plan, investigation, etc.)
│   └── User templates (tenant-configurable, future)
│
├── Context Assemblers (gather data for a template)
│   ├── Each template type has an assembler function
│   └── Assembler handles auth check: can this user see this data?
│
├── Renderers (produce output)
│   ├── PDFRenderer (ReportLab platypus — zero system deps, BSD)
│   ├── DOCXRenderer (python-docx)
│   └── HTMLRenderer (Django template rendering, always available)
│
└── Branding (tenant-aware headers/footers/logos)
    ├── Read from Tenant.settings["branding"] (existing field)
    └── Fallback to SVEND defaults
```

### Key Design Decisions

**1. Assemblers own authorization.** The render endpoint accepts a template name and entity IDs. The assembler for that template fetches the data AND verifies the requesting user can access it. No raw data passes through the API — only references.

**2. Renderers are swappable.** ReportLab is the engine. If we ever need to switch, only the renderer changes. Templates and assemblers are untouched.

**3. System templates are code. User templates are data.** System templates (CAPA format, A3 layout) are HTML files in the repo. User-configurable templates (future) are stored in the database with a template language TBD.

**4. Branding reads from Tenant model directly.** `Tenant.settings` is a JSONField that already exists. We read `settings.branding.company_name`, `settings.branding.logo_file_id`, `settings.branding.accent_color`. No TenantConfig dependency.

---

## Existing Code to Consolidate

| Current location | What it does | Lines | Pattern |
|-----------------|-------------|-------|---------|
| `a3_views.py:925-1030` | A3 report PDF | 105 | Markdown→HTML→render_to_string→WeasyPrint |
| `report_views.py:729-802` | Generic report PDF | 73 | Django template→WeasyPrint |
| `iso_doc_views.py:377-440` | ISO doc PDF | 63 | Django template→WeasyPrint |
| `iso_doc_views.py:449-520` | ISO doc DOCX | 71 | python-docx Document builder |
| `experimenter_views.py:2274+` | DOE run cards PDF | ~50 | Django template→WeasyPrint |
| `api/views.py:1271-1310` | Compliance page PDF | 39 | HTML→WeasyPrint with fallback |
| `whitepaper_views.py:67-95` | Whitepaper PDF | 28 | HTML→WeasyPrint |

**Common pattern across all 6 PDF sites:**
1. Fetch data and check auth
2. Render HTML via `render_to_string(template, context)`
3. Optionally process markdown
4. Load tenant branding (A3 does this, others don't)
5. `HTML(string=html_string).write_pdf(buffer)`
6. Return HttpResponse with content-disposition

This is exactly the pattern DocumentService encodes.

---

## Data Model

```python
# No new models needed for v1.
# Template registry is a Python module (system templates).
# Rendered documents are returned directly, not stored.
#
# Future: DocumentRender model for async generation + storage.
# Future: UserTemplate model for tenant-configurable templates.
```

---

## Template Registry

```python
# documents/registry.py

TEMPLATES = {
    # --- OLR-001 Knowledge Views ---
    "investigation_summary": {
        "label": "Investigation Summary",
        "description": "Full investigation with scoped subgraph, evidence, and conclusions",
        "assembler": "documents.assemblers.investigation_summary",
        "formats": ["pdf", "html"],
        "entity_type": "investigation",
    },
    "capa_compliance": {
        "label": "CAPA Compliance Report",
        "description": "Auto-assembled from investigation data in CAPA format for auditors",
        "assembler": "documents.assemblers.capa_compliance",
        "formats": ["pdf", "html"],
        "entity_type": "investigation",  # NOT capa — it's a VIEW on investigation
    },
    "control_plan": {
        "label": "Control Plan",
        "description": "What to monitor, how, how often — derived from knowledge structure",
        "assembler": "documents.assemblers.control_plan",
        "formats": ["pdf", "docx", "html"],
        "entity_type": "control_plan",
    },

    # --- Existing document types (consolidation) ---
    "a3_report": {
        "label": "A3 Report",
        "description": "One-page A3 problem solving report",
        "assembler": "documents.assemblers.a3_report",
        "formats": ["pdf", "html"],
        "entity_type": "a3_report",
    },
    "iso_document": {
        "label": "ISO Document",
        "description": "Controlled document with revision history",
        "assembler": "documents.assemblers.iso_document",
        "formats": ["pdf", "docx", "html"],
        "entity_type": "iso_document",
    },

    # --- Operational documents ---
    "doe_run_cards": {
        "label": "DOE Run Instruction Cards",
        "description": "Formatted cards for technicians: Run 1: A=-1, B=0, C=+1",
        "assembler": "documents.assemblers.doe_run_cards",
        "formats": ["pdf", "html"],
        "entity_type": "experiment_design",
    },
    "supplier_claim": {
        "label": "Supplier Claim Report",
        "description": "Claim lifecycle with CoA data and verification results",
        "assembler": "documents.assemblers.supplier_claim",
        "formats": ["pdf", "html"],
        "entity_type": "supplier_claim",
    },
    "eight_d": {
        "label": "8D Report",
        "description": "Eight Disciplines problem solving report",
        "assembler": "documents.assemblers.eight_d",
        "formats": ["pdf", "html"],
        "entity_type": "report",
    },

    # --- Knowledge health / leadership ---
    "knowledge_health": {
        "label": "Knowledge Health Report",
        "description": "Process knowledge metrics snapshot for management review",
        "assembler": "documents.assemblers.knowledge_health",
        "formats": ["pdf", "html"],
        "entity_type": "process_graph",
    },
}
```

### Extensibility

The registry is a dict. Adding a new document type = adding a key with an assembler path. User-configurable templates (future) would add entries dynamically from a database table, with assemblers that use a generic "fill fields from entity" pattern.

---

## API

### `POST /api/documents/render/`

```json
{
    "template": "a3_report",
    "entity_id": "uuid-of-a3-report",
    "format": "pdf",
    "options": {
        "include_diagrams": true,
        "force": false
    }
}
```

**Response:** Binary file (PDF/DOCX) or JSON with HTML string.

**Auth:** Session or API key. Assembler verifies entity access.

**Errors:**
- `400` — unknown template, unsupported format, missing entity_id
- `403` — user cannot access the referenced entity
- `404` — entity not found
- `500` — render failure (WeasyPrint error, etc.)

### `GET /api/documents/templates/`

Returns available templates for the user's tier. Team gets system templates. Enterprise gets system + user-configurable.

```json
{
    "templates": [
        {
            "key": "a3_report",
            "label": "A3 Report",
            "formats": ["pdf", "html"],
            "entity_type": "a3_report"
        }
    ]
}
```

---

## Service Class

```python
# documents/service.py

class DocumentService:
    """Unified document rendering service.

    Usage:
        result = DocumentService.render(
            template="a3_report",
            entity_id=uuid,
            format="pdf",
            user=request.user,
            options={}
        )
        # result.content_type = "application/pdf"
        # result.content = bytes
        # result.filename = "a3_report_title.pdf"
    """

    @classmethod
    def render(cls, template, entity_id, format, user, options=None):
        """Main entry point. Returns RenderResult."""

    @classmethod
    def get_branding(cls, user):
        """Load tenant branding for the user. Falls back to defaults."""

    @classmethod
    def available_templates(cls, user):
        """Templates available for this user's tier."""
```

---

## Assembler Pattern

Each assembler is a function that:
1. Takes `(entity_id, user, options)`
2. Verifies access (raises PermissionDenied if not)
3. Returns a context dict ready for template rendering

```python
# documents/assemblers.py

def a3_report(entity_id, user, options=None):
    """Assemble context for A3 report rendering."""
    from agents_api.models import A3Report
    from agents_api.permissions import qms_queryset

    qs, tenant, is_admin = qms_queryset(A3Report, user)
    report = qs.get(id=entity_id)  # Raises DoesNotExist → 404

    return {
        "report": report,
        "sections": _render_a3_sections(report),
        "project_title": report.project.title if report.project else "",
    }
```

---

## Renderer Pattern

```python
# documents/renderers.py

class PDFRenderer:
    """Render structured data to PDF via ReportLab platypus.

    Uses domain-specific document builders (A3Sheet, ControlPlanDoc, etc.)
    that compose ReportLab flowables. Each builder knows the layout for
    its document type — page size, margins, section structure, tables.

    The renderer handles common concerns: branding header/footer, page
    numbers, font registration, and the final PDF write.
    """

    @staticmethod
    def render(document_builder, branding):
        """Returns bytes. document_builder is a domain-specific builder instance."""
        from io import BytesIO
        from reportlab.platypus import SimpleDocTemplate

        buf = BytesIO()
        doc = SimpleDocTemplate(buf, **document_builder.page_config())
        flowables = document_builder.build(branding)
        doc.build(flowables, onFirstPage=_branded_header_footer(branding),
                  onLaterPages=_branded_header_footer(branding))
        return buf.getvalue()

class DOCXRenderer:
    """Render structured data to DOCX via python-docx."""

    @staticmethod
    def render(document_builder, branding):
        """Returns bytes."""
        from io import BytesIO
        from docx import Document

        doc = Document()
        document_builder.build_docx(doc, branding)
        buf = BytesIO()
        doc.save(buf)
        return buf.getvalue()

class HTMLRenderer:
    """Render HTML via Django templates — always available, no deps."""

    @staticmethod
    def render(template_path, context, branding):
        """Returns HTML string."""
        from django.template.loader import render_to_string
        context["branding"] = branding
        return render_to_string(template_path, context)
```

---

## File Structure

```
documents/
├── __init__.py
├── apps.py
├── registry.py          # Template definitions (TEMPLATES dict)
├── service.py           # DocumentService class
├── assemblers.py        # Context assembler functions (one per template type)
├── renderers.py         # PDFRenderer (ReportLab), DOCXRenderer, HTMLRenderer
├── builders/            # Domain-specific document builders (ReportLab platypus)
│   ├── __init__.py
│   ├── base.py          # BaseDocumentBuilder — shared branding, page config, fonts
│   ├── a3_sheet.py      # A3Sheet — landscape, 8-section grid layout
│   ├── investigation.py # InvestigationReport — multi-page, evidence tables
│   ├── control_plan.py  # ControlPlanDoc — tabular, derived from graph state
│   ├── capa.py          # CAPAComplianceDoc — auto-assembled from investigation
│   └── eight_d.py       # EightDReport — 8-section sequential
├── views.py             # API endpoints
├── urls.py              # URL routing
├── html_templates/      # Django templates for HTML-format output (web view)
│   ├── base_document.html
│   ├── a3_report.html
│   ├── investigation_summary.html
│   └── ...
└── tests.py
```

---

## Build Sequence

| # | Task | Est | Output |
|---|------|-----|--------|
| 1 | Create `documents/` app, register in settings | 15m | App skeleton |
| 2 | `builders/base.py` — BaseDocumentBuilder with ReportLab primitives, branding, fonts, page config | 2h | Foundation class |
| 3 | `renderers.py` — PDFRenderer (ReportLab), DOCXRenderer, HTMLRenderer | 1h | Three renderer classes |
| 4 | `registry.py` — template definitions | 30m | TEMPLATES dict |
| 5 | `service.py` — DocumentService with branding + Tempora async path | 1h | Main service class |
| 6 | `builders/a3_sheet.py` — first builder (landscape A3 layout) | 2h | Proof of concept |
| 7 | `assemblers.py` — first 3 assemblers (a3, investigation, control_plan) | 1.5h | Data pipelines |
| 8 | `builders/investigation.py` + `builders/control_plan.py` | 2h | Two more builders |
| 9 | `views.py` + `urls.py` — API endpoints | 45m | Live endpoints |
| 10 | `tests.py` — render each builder in each format | 1h | Verification |
| **Total** | | **~12h** | |

Note: More time than original estimate because we're building ReportLab document builders instead of HTML templates. The investment pays off — every builder is a reusable, testable, programmatic document definition. No CSS debugging, no pagination hacks.

---

## Open Questions for Eric

## Resolved Decisions

**OQ-1: PDF Engine → ReportLab.** DECIDED.

Evaluated 7 options (WeasyPrint, ReportLab, fpdf2, borb, pdfkit, xhtml2pdf, Typst). Full evaluation in session notes.

- WeasyPrint: good output, bad dep chain (cairo/pango). Every deployment fights it.
- borb: AGPL — disqualified for commercial product.
- pdfkit: dead upstream (wkhtmltopdf archived). Disqualified.
- xhtml2pdf: leaky ReportLab wrapper. If we need ReportLab anyway, use it directly.
- fpdf2: lightweight but hits walls on complex multi-page tables.
- Typst: excellent output, Python bindings too young for production.
- **ReportLab: BSD license, zero system deps, 25yr battle-tested, excellent output.** Winner.

Product opportunity: opinionated QMS document builder on ReportLab. Domain-specific primitives (A3Sheet, ControlPlan, InvestigationReport) wrapping ReportLab's platypus engine. Potentially standalone product.

**OQ-2: Async → Tempora from day 1.** DECIDED.

Synchronous for simple docs, Tempora job for large renders. Both paths through same service. The render endpoint returns either the document or a job ID based on estimated complexity.

**OQ-3: Storage → Return-only.** DECIDED.

Rendered documents are returned directly, not stored. Source data is the audit trail. Storage can be added later via a `DocumentRender` model if needed.

---

## Relationship to OLR-001

The standard brief makes clear: CAPA is a VIEW on investigation data, not a separate process. The Document Builder Service is how that view materializes. When an auditor asks "show me the CAPA for this investigation," the system calls:

```python
DocumentService.render(
    template="capa_compliance",
    entity_id=investigation.id,
    format="pdf",
    user=auditor,
)
```

The assembler pulls investigation data, evidence chain, root cause analysis, corrective actions, and verification results — all from the investigation and its graph connections. It formats them into the CAPA structure auditors expect. The investigation IS the CAPA. The document is the rendering.

Same pattern for management review (knowledge health report), control plans (graph filtered to monitored nodes), and every other compliance artifact OLR-001 defines as a view rather than a process.

---

*Spec ready for review. Awaiting Eric's decisions on OQ-1, OQ-2, OQ-3 before building.*
