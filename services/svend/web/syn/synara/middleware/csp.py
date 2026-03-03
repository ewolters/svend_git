"""
Content Security Policy (CSP) Middleware

SEC-001 §9.11 XSS Defense-in-Depth
Adds Content-Security-Policy header to all responses.

Compliance:
- SEC-001 §9.11: XSS Prevention
- OWASP CSP Cheat Sheet
- W3C Content Security Policy Level 3
"""

import logging

from django.conf import settings

logger = logging.getLogger(__name__)


class ContentSecurityPolicyMiddleware:
    """
    Add Content-Security-Policy header to HTTP responses.

    SEC-001 §9.11 XSS Defense-in-Depth:
    CSP restricts which resources can be loaded, providing defense-in-depth
    against XSS attacks even if other protections fail.

    Configuration in settings.py:
        CONTENT_SECURITY_POLICY = {
            "default-src": ["'self'"],
            "script-src": ["'self'"],
            ...
        }

    The middleware builds the header from the CONTENT_SECURITY_POLICY dict
    or uses the pre-built CSP_HEADER string if available.
    """

    def __init__(self, get_response):
        self.get_response = get_response
        self.csp_header = self._build_csp_header()
        if self.csp_header:
            logger.info(f"[CSP] Middleware initialized with policy: {self.csp_header[:100]}...")

    def _build_csp_header(self) -> str:
        """Build CSP header from settings."""
        # Use pre-built header if available
        if hasattr(settings, "CSP_HEADER") and settings.CSP_HEADER:
            return settings.CSP_HEADER

        # Build from dictionary
        csp_dict = getattr(settings, "CONTENT_SECURITY_POLICY", None)
        if not csp_dict:
            return ""

        return "; ".join(
            f"{directive} {' '.join(sources)}"
            for directive, sources in csp_dict.items()
        )

    def __call__(self, request):
        response = self.get_response(request)

        # Add CSP header if configured
        if self.csp_header:
            response["Content-Security-Policy"] = self.csp_header

        return response
