"""
Honeypot endpoints — traps for automated scanners.

These paths are never linked from the real application. Any request
to them is, by definition, hostile reconnaissance. Response:
1. Log to fail2ban action log → auto IP ban
2. Email alert
3. Tar pit the response (waste attacker time)
4. Return a convincing-looking fake response

Zero false positive rate: legitimate users never find these.
"""

import logging
import time

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .alerts import send_security_alert
from .bridge import log_security_action
from .cloudflare import block_ip

logger = logging.getLogger("syn.varta")


def _get_client_ip(request) -> str:
    cf_ip = request.META.get("HTTP_CF_CONNECTING_IP")
    if cf_ip:
        return cf_ip
    xff = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if xff:
        return xff.split(",")[-1].strip()
    return request.META.get("REMOTE_ADDR", "unknown")


def _honeypot_response(request, trap_name: str, response: HttpResponse) -> HttpResponse:
    """Common handler: log, alert, tar pit, respond."""
    ip = _get_client_ip(request)
    ua = request.META.get("HTTP_USER_AGENT", "")
    path = request.get_full_path()

    logger.warning(
        "HONEYPOT [%s] ip=%s ua=%s path=%s",
        trap_name,
        ip,
        ua,
        path,
    )

    # Log for fail2ban
    log_security_action(
        ip=ip,
        action="honeypot",
        detail=f"{trap_name} | ua={ua[:100]} | path={path[:200]}",
        score=10,
    )

    # Block at Cloudflare edge
    block_ip(ip, f"honeypot:{trap_name}")

    # Email alert — throttle by trap type only (not per-IP) to prevent storms
    send_security_alert(
        alert_type=f"honeypot_{trap_name}",
        subject=f"Honeypot hit: {trap_name}",
        body=(f"Trap: {trap_name}\nIP: {ip}\nPath: {path}\nUser-Agent: {ua}\nMethod: {request.method}\n"),
        severity="HIGH",
    )

    # Tar pit — make them wait
    time.sleep(5)

    return response


# ── Trap: WordPress admin ────────────────────────────────────────────


@csrf_exempt
def wp_admin(request):
    """Fake WordPress login page."""
    return _honeypot_response(
        request,
        "wp_admin",
        HttpResponse(
            "<html><head><title>Log In &lsaquo; Blog &#8212; WordPress</title></head>"
            "<body><div id='login'><h1><a href='#'>WordPress</a></h1>"
            "<form method='post'>"
            "<label>Username</label><input type='text' name='log'><br>"
            "<label>Password</label><input type='password' name='pwd'><br>"
            "<input type='submit' value='Log In'>"
            "</form></div></body></html>",
            content_type="text/html",
        ),
    )


# ── Trap: phpMyAdmin ─────────────────────────────────────────────────


@csrf_exempt
def phpmyadmin(request):
    """Fake phpMyAdmin page."""
    return _honeypot_response(
        request,
        "phpmyadmin",
        HttpResponse(
            "<html><head><title>phpMyAdmin</title></head>"
            "<body><h1>phpMyAdmin 5.2.1</h1>"
            "<form method='post'>"
            "<label>Username:</label><input type='text' name='pma_username'><br>"
            "<label>Password:</label><input type='password' name='pma_password'><br>"
            "<input type='submit' value='Go'>"
            "</form></body></html>",
            content_type="text/html",
        ),
    )


# ── Trap: .env file ──────────────────────────────────────────────────


@csrf_exempt
def fake_env(request):
    """Fake .env with canary credentials."""
    return _honeypot_response(
        request,
        "env_probe",
        HttpResponse(
            "# .env\n"
            "DB_HOST=localhost\n"
            "DB_USER=admin\n"
            "DB_PASSWORD=varta_canary_not_real_cred\n"
            "SECRET_KEY=varta_canary_this_is_a_trap\n"
            "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
            "AWS_SECRET_ACCESS_KEY=varta/canary/not/real\n"
            "STRIPE_SECRET_KEY=sk_live_varta_canary_fake\n",
            content_type="text/plain",
        ),
    )


# ── Trap: admin API ──────────────────────────────────────────────────


@csrf_exempt
def fake_admin_api(request):
    """Fake admin user list API."""
    return _honeypot_response(
        request,
        "admin_api",
        JsonResponse(
            {
                "users": [
                    {"id": 1, "username": "admin", "role": "superuser"},
                    {"id": 2, "username": "deploy", "role": "admin"},
                ],
                "total": 2,
                "page": 1,
            }
        ),
    )


# ── Trap: Git config ─────────────────────────────────────────────────


@csrf_exempt
def fake_git(request):
    """Fake .git/config — scanners love this one."""
    return _honeypot_response(
        request,
        "git_probe",
        HttpResponse(
            "[core]\n"
            "\trepositoryformatversion = 0\n"
            "\tfilemode = true\n"
            "\tbare = false\n"
            '[remote "origin"]\n'
            "\turl = git@github.com:varta-canary/honeypot.git\n"
            "\tfetch = +refs/heads/*:refs/remotes/origin/*\n",
            content_type="text/plain",
        ),
    )


# ── Trap: Backup files ───────────────────────────────────────────────


@csrf_exempt
def fake_backup(request):
    """Fake database backup endpoint."""
    return _honeypot_response(
        request,
        "backup_probe",
        HttpResponse(
            "-- PostgreSQL database dump (CANARY)\n-- This is not a real backup\n",
            content_type="text/plain",
        ),
    )
