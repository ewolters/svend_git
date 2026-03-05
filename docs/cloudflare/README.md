# Cloudflare — CDN, WAF, Tunnel

**Purpose:** Reverse proxy, DDoS protection, HTTPS termination, analytics.
**Dashboard:** https://dash.cloudflare.com
**Compliance:** SOC 2 Type II, ISO 27001

---

## How It Works

```
User → Cloudflare Edge (DDoS/WAF) → Cloudflare Tunnel → Caddy (localhost) → Gunicorn (127.0.0.1:8000)
```

No ports are exposed to the internet. Cloudflare Tunnel creates an outbound-only connection from the server to Cloudflare's edge network.

## Domain

- **Primary:** `svend.ai`
- **Alternate:** `www.svend.ai`
- **Server IP:** `108.193.83.15` (IPv4), `2600:1700:6d85:1000:e283:dff:8294:17ce` (IPv6)

## Services We Use

| Service | Purpose | Cost |
|---------|---------|------|
| Tunnel | Zero-trust reverse proxy (no open ports) | Free |
| WAF | Web application firewall, DDoS mitigation | Free tier |
| DNS | Authoritative DNS for svend.ai | Free |
| Analytics | JavaScript beacon for traffic analytics | Free |
| CDN | Edge caching of static assets | Free |

## Integration Points

| File | Purpose |
|------|---------|
| `Caddyfile` | Local reverse proxy (Tunnel terminates here) |
| `svend/settings.py` | CSP headers include `static.cloudflareinsights.com` |
| `api/landing_views.py` | `HTTP_CF_IPCOUNTRY` header for regional pricing |
| `templates/` | Analytics beacon script (token: `322c2cefd22b4002b8aecf303abbb639`) |

## Headers We Rely On

| Header | Purpose |
|--------|---------|
| `CF-IPCountry` | Geo-locate visitor for regional pricing |
| `CF-Connecting-IP` | Real client IP (behind proxy) |
| `X-Forwarded-For` | Proxy chain (DRF NUM_PROXIES=1) |

## Caddy Configuration

Caddy sits between the Cloudflare Tunnel and Gunicorn. It handles:
- Security headers (HSTS 2yr preload, X-Frame-Options: DENY, nosniff, Referrer-Policy)
- JSON access logging to `/var/log/caddy/svend.log`
- Static file pass-through

## Tunnel Management

The tunnel runs as a `cloudflared` service. Common commands:

```bash
# Check tunnel status
systemctl status cloudflared

# Restart tunnel
sudo systemctl restart cloudflared

# View tunnel logs
journalctl -u cloudflared -n 50
```

## DNS Records

Managed in Cloudflare Dashboard → DNS. The tunnel creates a CNAME record that routes `svend.ai` through Cloudflare's edge.

## Security Headers (set by Caddy)

- `Strict-Transport-Security: max-age=63072000; includeSubDomains; preload`
- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: strict-origin-when-cross-origin`
- CSP configured in Django `settings.py`
