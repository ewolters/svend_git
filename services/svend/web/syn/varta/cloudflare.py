"""
Cloudflare firewall integration for Varta.

Blocks attacker IPs at the Cloudflare edge using the Access Rules API,
so malicious requests never reach the origin server.

Requires two env vars:
  CLOUDFLARE_API_TOKEN  — scoped to Zone.Firewall Services (edit)
  CLOUDFLARE_ZONE_ID    — zone ID for svend.ai

Uses Access Rules (IP block) rather than WAF custom rules because:
- Simpler (one API call per block)
- No rule count limits for IP access rules
- Auto-expires via the notes field + periodic cleanup
"""

import json
import logging
import os
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("syn.varta")

_CF_API = "https://api.cloudflare.com/client/v4"
_blocked_cache: set[str] = set()  # Avoid duplicate API calls
_MAX_CACHE = 5000


def _get_config() -> tuple[str, str] | None:
    """Return (token, zone_id) or None if not configured."""
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    zone_id = os.environ.get("CLOUDFLARE_ZONE_ID")
    if not token or not zone_id:
        return None
    return token, zone_id


def block_ip(ip: str, reason: str = "Varta auto-block") -> bool:
    """
    Block an IP at the Cloudflare edge.

    Returns True if blocked (or already blocked), False on failure.
    """
    if ip in _blocked_cache:
        return True

    config = _get_config()
    if config is None:
        logger.debug("Cloudflare not configured — skipping edge block for %s", ip)
        return False

    token, zone_id = config

    # Determine IP version for the access rule
    mode = "block"
    target = "ip"
    if ":" in ip:
        target = "ip6"

    payload = json.dumps(
        {
            "mode": mode,
            "configuration": {
                "target": target,
                "value": ip,
            },
            "notes": f"Varta: {reason} | {time.strftime('%Y-%m-%d %H:%M:%S')}",
        }
    ).encode()

    url = f"{_CF_API}/zones/{zone_id}/firewall/access_rules/rules"
    req = Request(url, data=payload, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")

    try:
        with urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            if body.get("success"):
                _blocked_cache.add(ip)
                if len(_blocked_cache) > _MAX_CACHE:
                    _blocked_cache.clear()
                logger.info("Cloudflare blocked IP: %s", ip)
                return True
            else:
                errors = body.get("errors", [])
                # "firewallaccessrules.api.duplicate_of_existing" means already blocked
                for err in errors:
                    if "duplicate" in str(err.get("message", "")).lower():
                        _blocked_cache.add(ip)
                        return True
                logger.warning("Cloudflare block failed for %s: %s", ip, errors)
                return False
    except URLError as e:
        logger.warning("Cloudflare API error blocking %s: %s", ip, e)
        return False
