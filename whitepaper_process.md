# Whitepaper Publishing Process

How to go from a written whitepaper to a live, downloadable PDF on svend.ai.

## Prerequisites

- WhitePaper model already exists (`api/models.py`) with fields: `title`, `slug`, `description`, `body`, `meta_description`, `topic`, `status`, `gated`, `published_at`
- WeasyPrint installed system-wide (`pip3 install weasyprint`)
- Routes live at `/whitepapers/`, `/whitepapers/<slug>/`, `/whitepapers/<slug>/pdf/`

## Step 1: Write the Whitepaper

Write the whitepaper as a standalone HTML document. Use the SEO strategy doc (`svend-seo-strategy.html`) or the first whitepaper (`svend-whitepaper-insight-spine.html`) as reference for tone, structure, and styling.

Structure:
- **Header**: Logo, "Whitepaper" label, title, subtitle, meta (date, read time)
- **Sections**: Numbered (`Section 01`, `Section 02`, ...), each with h2 + content
- **Elements**: Pull quotes, stat callout rows, contrast tables, cards (green/gold/blue/orange), dividers
- **Interactive placeholders**: Mark with `<div class="chart-placeholder">` and a label describing what the interactive should do
- **CTA**: At the end, linking to svend.ai
- **Footer**: Logo, copyright

## Step 2: Build Interactive Sections

Replace each `chart-placeholder` with a working interactive. Design them to:

1. **Work without JS** — CSS renders a clean static version (this is what appears in the PDF)
2. **Enhance with JS** — Animations, click handlers, sliders, calculations
3. **Print gracefully** — Add `@media print` overrides if needed (hide buttons, force visibility)

Wrap all JS in an IIFE at the bottom of the body content:
```html
<script>
(function() {
    // Interactive code here
})();
</script>
```

## Step 3: Extract Body Content

The database stores the **body only** — not the full HTML document. Extract:

1. **CSS**: All whitepaper-specific styles (section-num, pull-quote, stat-row, spine-diagram, contrast-table, cards, interactive styles). Put in a `<style>` block. Use CSS variables (`var(--accent-primary)`, etc.) — they're defined by both the web template and the print template.

2. **Article HTML**: Everything between the header and footer/CTA of the original document. Wrap in `<div class="wp-article">`.

3. **Scripts**: Any interactive JS, wrapped in `<script>` tags at the end.

The body content file lives at:
```
templates/whitepapers/_<slug>_body.html
```

This is a reference copy — the actual content is stored in the database.

## Step 4: Create the Database Entry

From the Django project directory (`services/svend/web/`):

```bash
source ~/Desktop/svend_transfer/k/kjerne/.venv/bin/activate
DJANGO_SETTINGS_MODULE=svend.settings python << 'EOF'
import django; django.setup()
from django.utils import timezone
from api.models import WhitePaper

with open('templates/whitepapers/_<slug>_body.html', 'r') as f:
    body = f.read()

wp, created = WhitePaper.objects.update_or_create(
    slug='<slug>',
    defaults={
        'title': '<Full Title>',
        'description': '<1-2 sentence abstract for the index page>',
        'body': body,
        'meta_description': '<Under 160 chars for SEO>',
        'topic': '<Topic Tag>',
        'status': 'published',
        'published_at': timezone.now(),
    }
)
print(f"{'Created' if created else 'Updated'}: {wp.id} | {wp.slug}")
EOF
```

**Constraints**:
- `meta_description`: max 160 characters (DB enforced)
- `slug`: max 200 characters, auto-generated from title if blank
- `body`: unlimited text

## Step 5: Reload and Test

```bash
# Reload gunicorn
kill -HUP $(pgrep -f 'gunicorn' | head -1)

# Test all three routes
curl -s -o /dev/null -w "%{http_code}" -H "X-Forwarded-Proto: https" \
    http://127.0.0.1:8000/whitepapers/
# → 200 (index lists the paper)

curl -s -o /dev/null -w "%{http_code}" -H "X-Forwarded-Proto: https" \
    http://127.0.0.1:8000/whitepapers/<slug>/
# → 200 (web detail page)

curl -s -o /tmp/test.pdf -w "%{http_code}" -H "X-Forwarded-Proto: https" \
    http://127.0.0.1:8000/whitepapers/<slug>/pdf/
# → 200 (PDF download)

# Verify PDF
file /tmp/test.pdf        # → PDF document, version 1.7
pdfinfo /tmp/test.pdf     # → Pages, size, title
pdftotext /tmp/test.pdf - | head -20  # → Spot-check text extraction

# Verify sitemap
curl -s -H "X-Forwarded-Proto: https" \
    http://127.0.0.1:8000/sitemap.xml | grep '<slug>'
```

## Step 6: Update the Whitepaper (if needed)

To update body content after publishing:

```bash
DJANGO_SETTINGS_MODULE=svend.settings python << 'EOF'
import django; django.setup()
from api.models import WhitePaper

wp = WhitePaper.objects.get(slug='<slug>')
with open('templates/whitepapers/_<slug>_body.html', 'r') as f:
    wp.body = f.read()
wp.save()
print(f"Updated: {wp.slug}")
EOF
```

Then reload gunicorn — PDF regenerates on every request from the current body.

## Architecture Notes

**Web detail page** (`whitepaper_detail.html`):
- Extends `tool_base.html` (Svend nav, footer, dark theme CSS variables)
- Header section: title, description, topic tag, date, PDF download button
- Body rendered with `{{ paper.body|safe }}` — the body's own `<style>` block takes effect
- Markdown fallback: if body doesn't start with `<style`/`<div`/`<span`, marked.js renders it as markdown

**PDF print template** (`whitepaper_print.html`):
- Standalone HTML (no Django template inheritance)
- Defines CSS variables mapped to light-theme values (white background, dark text)
- A4 page size, 2.5cm/2cm margins
- Page numbers (`page / pages`) at bottom center, `svend.ai` at bottom right
- Cover page (title, topic, date, Svend branding) with `page-break-after: always`
- Back cover (logo, tagline, URL) with `page-break-before: always`
- `{{ paper.body|safe }}` renders between cover and back cover

**Analytics**:
- Every web view and PDF download creates a `WhitePaperDownload` record
- Tracks: referrer domain, IP hash, user agent, bot detection
- Viewable via internal dashboard at `/api/internal/whitepapers/analytics/`

## File Inventory

```
api/whitepaper_views.py              # Public views (list, detail, pdf)
api/models.py                        # WhitePaper, WhitePaperDownload models
svend/urls.py                        # Routes + WhitePaperSitemap
templates/whitepapers.html           # Index page (lists all published)
templates/whitepaper_detail.html     # Web reader (extends tool_base.html)
templates/whitepaper_print.html      # PDF template (WeasyPrint)
templates/whitepapers/_*_body.html   # Reference copies of body content
```
