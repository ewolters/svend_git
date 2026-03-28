"""
Request threat scoring engine.

Assigns a numeric threat score to each request based on signals:
- Path traversal patterns
- SQL injection fragments
- Known scanner User-Agents
- Excessive error rate from same IP
- Suspicious headers

Score thresholds:
  0-2   : Normal traffic
  3-5   : Suspicious — log, monitor
  6-9   : Likely hostile — tar pit
  10+   : Confirmed hostile — ban via fail2ban
"""

import re
import time
from collections import defaultdict

# ── Pattern libraries ────────────────────────────────────────────────

PATH_TRAVERSAL = re.compile(r"(\.\./|\.\.\\|%2e%2e|%252e|%c0%ae|%c1%9c)", re.IGNORECASE)

# SQL injection — only multi-keyword patterns to reduce false positives.
# Single quotes/semicolons in POST bodies are normal form data.
SQL_INJECTION = re.compile(
    r"(\bunion\s+(all\s+)?select\b|\bselect\s+.*\bfrom\b.*\bwhere\b"
    r"|\b(drop|alter)\s+(table|database|index)\b"
    r"|;\s*(drop|delete|update|insert)\b"
    r"|\bexec(\s+|\()xp_|\bbenchmark\s*\(|\bsleep\s*\(\d"
    r"|\bwaitfor\s+delay\b|'\s*(or|and)\s+['\d])",
    re.IGNORECASE,
)

XSS_PATTERNS = re.compile(
    r"(<script|javascript:|on(error|load|click|mouseover)\s*=|<img[^>]+onerror)",
    re.IGNORECASE,
)

COMMAND_INJECTION = re.compile(
    r"(;\s*(ls|cat|whoami|id|pwd|wget|curl|nc|bash|sh|python|perl|ruby)\b"
    r"|\|\s*(ls|cat|whoami|id)|`[^`]+`|\$\(.*\))",
    re.IGNORECASE,
)

SCANNER_USER_AGENTS = re.compile(
    r"(nikto|sqlmap|nmap|masscan|zgrab|gobuster|dirbuster|wfuzz|hydra"
    r"|burpsuite|acunetix|nessus|openvas|whatweb|nuclei|httpx"
    r"|python-requests/|Go-http-client|libwww-perl|curl/|wget/)",
    re.IGNORECASE,
)

# Paths that only scanners hit — scored separately from honeypots
SCANNER_PATHS = re.compile(
    r"^/(\.git/|\.svn/|\.hg/|\.env\b|wp-content/|wp-includes/"
    r"|cgi-bin/|\.well-known/security\.txt"
    r"|vendor/phpunit|telescope/|_debugbar|__debug__"
    r"|actuator/|\.DS_Store|thumbs\.db|web\.config"
    r"|server-status|server-info)",
    re.IGNORECASE,
)

# ── Per-IP error tracking ────────────────────────────────────────────

# Simple in-memory tracker: {ip: [(timestamp, status_code), ...]}
# Cleared periodically to avoid memory growth
_error_history: dict[str, list[tuple[float, int]]] = defaultdict(list)
_last_cleanup = time.time()
ERROR_WINDOW = 300  # 5-minute window
ERROR_THRESHOLD = 15  # 15 errors in window = suspicious
CLEANUP_INTERVAL = 600  # Clean up every 10 minutes


def _cleanup_errors():
    """Remove stale entries to prevent memory growth."""
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < CLEANUP_INTERVAL:
        return
    _last_cleanup = now
    cutoff = now - ERROR_WINDOW
    stale_ips = []
    for ip, entries in _error_history.items():
        _error_history[ip] = [(t, s) for t, s in entries if t > cutoff]
        if not _error_history[ip]:
            stale_ips.append(ip)
    for ip in stale_ips:
        del _error_history[ip]


def record_error(ip: str, status_code: int):
    """Record a 4xx/5xx response for an IP."""
    _error_history[ip].append((time.time(), status_code))
    _cleanup_errors()


def get_error_rate(ip: str) -> int:
    """Count recent errors from an IP."""
    cutoff = time.time() - ERROR_WINDOW
    return sum(1 for t, _ in _error_history.get(ip, []) if t > cutoff)


# ── Scoring engine ───────────────────────────────────────────────────


def score_request(request) -> tuple[int, list[str]]:
    """
    Score a request for threat indicators.

    Returns (score, reasons) where reasons is a list of triggered rules.
    """
    score = 0
    reasons = []

    path = request.path
    query = request.META.get("QUERY_STRING", "")
    ua = request.META.get("HTTP_USER_AGENT", "")
    full_url = f"{path}?{query}" if query else path

    # ── Path analysis ────────────────────────────────────────────
    if PATH_TRAVERSAL.search(full_url):
        score += 8
        reasons.append("path_traversal")

    if SCANNER_PATHS.match(path):
        score += 6
        reasons.append("scanner_path")

    # ── Query/body injection ─────────────────────────────────────
    check_str = (
        f"{full_url} {request.body.decode('utf-8', errors='ignore')[:2000]}" if request.method == "POST" else full_url
    )

    if SQL_INJECTION.search(check_str):
        score += 7
        reasons.append("sql_injection")

    if XSS_PATTERNS.search(check_str):
        score += 6
        reasons.append("xss_attempt")

    if COMMAND_INJECTION.search(check_str):
        score += 8
        reasons.append("command_injection")

    # ── User-Agent ───────────────────────────────────────────────
    if SCANNER_USER_AGENTS.search(ua):
        score += 5
        reasons.append("scanner_ua")

    if not ua or len(ua) < 10:
        score += 2
        reasons.append("missing_ua")

    # ── Error rate ───────────────────────────────────────────────
    ip = _get_client_ip(request)
    error_count = get_error_rate(ip)
    if error_count >= ERROR_THRESHOLD:
        score += 4
        reasons.append(f"error_rate:{error_count}")

    # ── Suspicious headers ───────────────────────────────────────
    # X-Forwarded-For spoofing (multiple entries beyond Cloudflare's 1)
    xff = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if xff.count(",") > 2:
        score += 3
        reasons.append("xff_spoofing")

    return score, reasons


def _get_client_ip(request) -> str:
    """Extract client IP, accounting for Cloudflare proxy."""
    # CF-Connecting-IP is authoritative behind Cloudflare
    cf_ip = request.META.get("HTTP_CF_CONNECTING_IP")
    if cf_ip:
        return cf_ip
    xff = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if xff:
        # Rightmost entry is what Cloudflare appends
        return xff.split(",")[-1].strip()
    return request.META.get("REMOTE_ADDR", "unknown")
