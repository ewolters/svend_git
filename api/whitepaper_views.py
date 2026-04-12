"""Public whitepaper views — no auth required, SEO-optimized with PDF download."""

import hashlib
import re
from io import BytesIO
from urllib.parse import urlparse

from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
from django.template.loader import render_to_string

from .models import WhitePaper, WhitePaperDownload

BOT_PATTERN = re.compile(
    r"bot|crawl|spider|slurp|bingpreview|facebookexternalhit|Googlebot|"
    r"Baiduspider|YandexBot|DuckDuckBot|Twitterbot|LinkedInBot",
    re.IGNORECASE,
)


def _record_download(request, paper):
    """Record a whitepaper view/download for analytics."""
    ua = request.META.get("HTTP_USER_AGENT", "")
    referrer = request.META.get("HTTP_REFERER", "")
    ip = request.META.get("HTTP_CF_CONNECTING_IP", "") or request.META.get("REMOTE_ADDR", "")
    ip_hash = hashlib.sha256(ip.encode()).hexdigest() if ip else ""

    ref_domain = ""
    if referrer:
        try:
            ref_domain = urlparse(referrer).netloc
        except Exception:
            pass

    is_bot = bool(BOT_PATTERN.search(ua))

    WhitePaperDownload.objects.create(
        paper=paper,
        referrer_domain=ref_domain[:200],
        ip_hash=ip_hash,
        user_agent=ua[:500],
        is_bot=is_bot,
    )


def whitepaper_list(request):
    """Published whitepapers, newest first."""
    papers = WhitePaper.objects.filter(status=WhitePaper.Status.PUBLISHED).order_by("-published_at")
    return render(request, "whitepapers.html", {"papers": papers})


def whitepaper_detail(request, slug):
    """Single whitepaper — web-readable view with SEO metadata."""
    paper = get_object_or_404(WhitePaper, slug=slug, status=WhitePaper.Status.PUBLISHED)
    try:
        _record_download(request, paper)
    except Exception:
        pass
    return render(request, "whitepaper_detail.html", {"paper": paper})


def whitepaper_pdf(request, slug):
    """Generate and serve whitepaper as PDF via WeasyPrint."""
    paper = get_object_or_404(WhitePaper, slug=slug, status=WhitePaper.Status.PUBLISHED)
    try:
        _record_download(request, paper)
    except Exception:
        pass

    # Render print-optimized HTML
    html_string = render_to_string("whitepaper_print.html", {"paper": paper})

    # Generate PDF
    from weasyprint import HTML

    pdf_file = BytesIO()
    HTML(string=html_string, base_url="https://svend.ai").write_pdf(pdf_file)
    pdf_file.seek(0)

    response = HttpResponse(pdf_file.read(), content_type="application/pdf")
    safe_slug = re.sub(r"[^\w\-.]", "_", paper.slug) or "whitepaper"
    response["Content-Disposition"] = f'inline; filename="{safe_slug}.pdf"'
    return response
