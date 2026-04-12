"""
Варта middleware — the main active defense layer.

Sits early in the middleware stack. For each request:
1. Check if IP is in the tar pit list → slow response
2. Score the request for threat indicators
3. If score >= BAN_THRESHOLD → log for fail2ban, alert
4. If score >= TARPIT_THRESHOLD → add IP to tar pit
5. After response: track error rates per IP

Thread-safe, no database queries, minimal overhead for clean traffic.
"""

import logging
import time

from django.http import HttpResponse

from .alerts import send_security_alert
from .bridge import log_security_action
from .cloudflare import block_ip
from .scoring import record_error, score_request

logger = logging.getLogger("syn.varta")

# ── Thresholds ───────────────────────────────────────────────────────
TARPIT_THRESHOLD = 6  # Score >= 6: slow them down
BAN_THRESHOLD = 10  # Score >= 10: log for fail2ban auto-ban
TARPIT_SECONDS = 8  # How long to delay tar-pitted requests

# ── Tar pit tracking ────────────────────────────────────────────────
# In-memory set of IPs currently being tar-pitted
# {ip: expiry_timestamp}
_tarpit_ips: dict[str, float] = {}
TARPIT_DURATION = 3600  # Keep IP in tar pit for 1 hour

# ── Allowlist ────────────────────────────────────────────────────────
# IPs that should never be scored/blocked
ALLOWLIST = frozenset(
    {
        "127.0.0.1",
        "::1",
    }
)

# Cloudflare IP ranges (proxy IPs — these appear as REMOTE_ADDR)
# Source: https://www.cloudflare.com/ips/
# We score the CLIENT IP (CF-Connecting-IP), not the proxy IP.
# But if CF-Connecting-IP is missing, REMOTE_ADDR is the Cloudflare proxy.
CLOUDFLARE_V4_PREFIXES = (
    "173.245.48.",
    "103.21.244.",
    "103.22.200.",
    "103.31.4.",
    "141.101.",
    "108.162.",
    "190.93.",
    "188.114.",
    "197.234.",
    "198.41.",
    "162.158.",
    "104.16.",
    "104.17.",
    "104.18.",
    "104.19.",
    "104.20.",
    "104.21.",
    "104.22.",
    "104.23.",
    "104.24.",
    "104.25.",
    "104.26.",
    "104.27.",
    "172.64.",
    "131.0.72.",
)


def _get_client_ip(request) -> str:
    """Extract the real client IP, not Cloudflare's proxy IP."""
    cf_ip = request.META.get("HTTP_CF_CONNECTING_IP")
    if cf_ip:
        return cf_ip
    xff = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if xff:
        return xff.split(",")[-1].strip()
    return request.META.get("REMOTE_ADDR", "unknown")


def _is_allowlisted(ip: str) -> bool:
    """Check if IP should bypass all scoring."""
    if ip in ALLOWLIST:
        return True
    # Tailscale IPs (100.64.0.0/10)
    if ip.startswith("100."):
        try:
            second_octet = int(ip.split(".")[1])
            if 64 <= second_octet <= 127:
                return True
        except (IndexError, ValueError):
            pass
    # Cloudflare proxy IPs — don't ban the CDN
    if any(ip.startswith(prefix) for prefix in CLOUDFLARE_V4_PREFIXES):
        return True
    return False


def _is_tarpitted(ip: str) -> bool:
    """Check if IP is in the tar pit (and clean expired entries)."""
    now = time.time()
    expiry = _tarpit_ips.get(ip)
    if expiry is None:
        return False
    if now > expiry:
        del _tarpit_ips[ip]
        return False
    return True


def _add_to_tarpit(ip: str):
    """Add an IP to the tar pit."""
    _tarpit_ips[ip] = time.time() + TARPIT_DURATION
    # Prevent unbounded growth: cap at 10000 entries
    if len(_tarpit_ips) > 10000:
        # Remove oldest entries
        now = time.time()
        expired = [k for k, v in _tarpit_ips.items() if v < now]
        for k in expired:
            del _tarpit_ips[k]


class VartaMiddleware:
    """
    Active defense middleware.

    Must be placed early in the stack — after SecurityMiddleware
    but before session/auth middleware to block threats before
    wasting resources on request processing.
    """

    def __init__(self, get_response):
        self.get_response = get_response
        logger.info("[ВАРТА] Active defense middleware initialized")

    def __call__(self, request):
        ip = _get_client_ip(request)

        # ── Skip allowlisted IPs ─────────────────────────────────
        if _is_allowlisted(ip):
            return self.get_response(request)

        # ── Tar pit check (before scoring — already flagged) ─────
        if _is_tarpitted(ip):
            # Slow them down significantly
            time.sleep(TARPIT_SECONDS)
            logger.debug("Tar-pitted request from %s", ip)

        # ── Score the request ────────────────────────────────────
        threat_score, reasons = score_request(request)

        if threat_score >= BAN_THRESHOLD:
            # Nuclear option: log for fail2ban + alert + tar pit
            reason_str = ",".join(reasons)
            log_security_action(
                ip=ip,
                action="ban",
                detail=f"score={threat_score} reasons={reason_str} path={request.path[:200]}",
                score=threat_score,
            )
            send_security_alert(
                alert_type="high_threat",
                subject=f"High threat request blocked (score {threat_score})",
                body=(
                    f"IP: {ip}\n"
                    f"Score: {threat_score}\n"
                    f"Reasons: {reason_str}\n"
                    f"Path: {request.get_full_path()[:500]}\n"
                    f"User-Agent: {request.META.get('HTTP_USER_AGENT', 'none')}\n"
                    f"Method: {request.method}\n"
                ),
                severity="CRITICAL",
            )
            _add_to_tarpit(ip)
            block_ip(ip, f"score={threat_score} {reason_str}")

            # Return a boring 404 — don't reveal we caught them
            time.sleep(TARPIT_SECONDS)
            return HttpResponse("Not Found", status=404, content_type="text/plain")

        elif threat_score >= TARPIT_THRESHOLD:
            # Suspicious but not definitive — tar pit + log
            reason_str = ",".join(reasons)
            log_security_action(
                ip=ip,
                action="tarpit",
                detail=f"score={threat_score} reasons={reason_str}",
                score=threat_score,
            )
            _add_to_tarpit(ip)
            time.sleep(TARPIT_SECONDS)

        # ── Attach score to request for downstream use ───────────
        request.varta_score = threat_score
        request.varta_reasons = reasons

        # ── Process request normally ─────────────────────────────
        response = self.get_response(request)

        # ── Post-response: track errors ──────────────────────────
        if response.status_code >= 400:
            record_error(ip, response.status_code)

        return response
