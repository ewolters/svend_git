"""Custom authentication for API endpoints."""

from rest_framework.authentication import SessionAuthentication


class CsrfExemptSessionAuthentication(SessionAuthentication):
    """Session authentication without CSRF enforcement.

    For API endpoints accessed via JavaScript fetch with credentials,
    the session cookie provides authentication. CSRF is not needed
    since we're not using form submissions.
    """

    def enforce_csrf(self, request):
        """Skip CSRF check for API endpoints."""
        return None
