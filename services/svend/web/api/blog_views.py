"""Public blog views — no auth required, SEO-optimized."""

import hashlib
import re
from urllib.parse import urlparse

from django.shortcuts import get_object_or_404, render

from .models import BlogPost, BlogView

BOT_PATTERN = re.compile(
    r"bot|crawl|spider|slurp|bingpreview|facebookexternalhit|Googlebot|"
    r"Baiduspider|YandexBot|DuckDuckBot|Twitterbot|LinkedInBot",
    re.IGNORECASE,
)


def _record_view(request, post):
    """Record a blog view asynchronously-safe (fire and forget)."""
    ua = request.META.get("HTTP_USER_AGENT", "")
    referrer = request.META.get("HTTP_REFERER", "")
    ip = request.META.get("HTTP_X_FORWARDED_FOR", "").split(",")[0].strip() or request.META.get("REMOTE_ADDR", "")

    # Hash IP for privacy
    ip_hash = hashlib.sha256(ip.encode()).hexdigest() if ip else ""

    # Extract referrer domain
    ref_domain = ""
    if referrer:
        try:
            ref_domain = urlparse(referrer).netloc
        except Exception:
            pass

    is_bot = bool(BOT_PATTERN.search(ua))

    BlogView.objects.create(
        post=post,
        referrer=referrer[:500],
        referrer_domain=ref_domain[:200],
        path=request.get_full_path()[:300],
        ip_hash=ip_hash,
        user_agent=ua[:500],
        is_bot=is_bot,
    )


def blog_list(request):
    """Published blog posts, newest first."""
    posts = BlogPost.objects.filter(status=BlogPost.Status.PUBLISHED).order_by(
        "-published_at"
    )
    return render(request, "blog_list.html", {"posts": posts})


def blog_detail(request, slug):
    """Single blog post with Article schema markup."""
    post = get_object_or_404(BlogPost, slug=slug, status=BlogPost.Status.PUBLISHED)
    try:
        _record_view(request, post)
    except Exception:
        pass  # Never let analytics break the page
    return render(request, "blog_detail.html", {"post": post})
