# Security Debt — Hardening Audit 2026-02-26

Threat model: nation-state (RU APT), motivated by pro-Ukraine content.
Audit scope: full codebase at `/home/eric/kjerne/services/svend/web/`.

## Resolved — 2026-02-26

### CRITICAL

- [x] **C1 — RCE via exec() in execute_code** | `agents_api/dsw/endpoints_data.py:165-258`
  Builtins restriction was bypassable (`pd.__builtins__['__import__']('os').system(...)`).
  Any authenticated user (incl. free tier) got full shell.
  **Fix:** Endpoint disabled (returns 403). Needs container sandbox before re-enabling.

- [x] **C2 — Rate limit bypass via X-Forwarded-For spoofing** | `settings.py`, `middleware.py`, `views.py`, `whitepaper_views.py`, `blog_views.py`
  `NUM_PROXIES` was not set — DRF used client-controlled leftmost XFF entry.
  All anonymous rate limits (login 5/min, register 5/hr) were bypassable.
  **Fix:** Added `NUM_PROXIES: 1` to DRF config. Switched all IP extraction to `CF-Connecting-IP`.

### HIGH

- [x] **H1 — Monte Carlo eval escape via np module** | `agents_api/dsw_views.py:2846-2876`
  `np` in AST allowed names + only filtering `ast.Call(ast.Attribute(...))` left bare
  attribute chains exploitable: `np.__class__.__init__.__globals__[...]`.
  **Fix:** Removed `np` from allowed_names, blocked ALL `ast.Attribute` nodes, added
  `mean`/`std`/`sum` as direct functions. Now matches the already-fixed `dsw/simulation.py`.

## Open — Prioritized

### P0 — Fix This Week

| ID | Finding | File(s) | Impact |
|----|---------|---------|--------|
| H2 | `pickle.load()` on user model files — RCE if filesystem write is chained | `endpoints_ml.py:477,590`, `dsw_views.py:2838` | Persistent RCE |
| H3 | No per-account lockout — distributed brute-force unlimited | cross-cutting | Account takeover |
| H4 | Username/email enumeration via registration (distinct error msgs) | `views.py:876-886` | Recon |
| H5 | Login timing side-channel — 2nd bcrypt call only if email exists | `views.py:704-714` | Email enumeration |
| H6 | Silent decryption failure returns ciphertext as plaintext | `encryption.py:51-55` | Data leak |
| H7 | No webhook idempotency — Stripe events can be replayed | `billing.py:471-548` | Billing manipulation |
| H8 | Subscription sync race condition — no locking | `billing.py:276-337` | Tier escalation |

### P1 — Fix During Overhaul

| ID | Finding | File(s) | Impact |
|----|---------|---------|--------|
| M1 | No session timeout (default 14 days) | `settings.py` | Stolen session longevity |
| M2 | PBKDF2 default — should be Argon2 for GPU resistance | `settings.py` | Offline cracking |
| M3 | Password change doesn't kill other sessions | `views.py:961-966` | Compromised session persists |
| M4 | Guest tokens bypass rate limiting entirely | `permissions.py:234-255` | DoS via guest endpoints |
| M5 | TOCTOU race in can_query() + rate_limited decorator | `models.py:223-235`, `permissions.py:53-69` | Limit bypass |
| M6 | InviteCode 32-bit entropy + race condition (double-spend) | `models.py:109-124` | Code brute-force |
| M7 | Email verification tokens never expire | `models.py:246-254` | Stale token reuse |
| M8 | HTML injection in PDF title (wkhtmltopdf SSRF to internal net) | `views.py:1106` | SSRF |
| M9 | Regex HTML sanitizer bypassable (img/svg/body event handlers) | `views.py:1087-1099` | XSS in PDF |
| M10 | Tempora binds to 0.0.0.0 — cluster protocol on public IP | `tempora/coordination/server.py:109` | Unauthenticated access |
| M11 | Service runs as user eric (same as code/key owner) | `svend.service:7-8` | Blast radius |
| M12 | Disabled account returns distinct 403 (confirms valid creds) | `views.py:716-727` | Account enumeration |
| M13 | Django admin at default /admin/ — no proxy rate limiting | `urls.py:158` | Credential stuffing target |
| M14 | CSP allows unsafe-inline + unsafe-eval (XSS protection is zero) | `Caddyfile:25` | XSS |
| M15 | Empty SECRET_KEY default — no startup validation | `config.py:28` | Total compromise if env fails |
| M16 | Founder slot check TOCTOU race | `billing.py:374-377` | Oversold slots |
| M17 | Non-constant-time token comparison in verify_email | `models.py:285-292` | Timing attack |
| M18 | Guest token stored plaintext (not hashed like verification tokens) | `permissions.py:238`, `models.py` | DB leak → instant access |
| M19 | Unsalted SHA-256 for token hashing (no HMAC key) | `encryption.py:78-79` | Rainbow tables |

### P2 — Hardening (Lower Priority)

| ID | Finding | Impact |
|----|---------|--------|
| L1 | PII (email) logged in login failure messages | GDPR compliance |
| L2 | Internal exception details in PDF error + enterprise model error | Info leakage |
| L3 | Tier info disclosed in 429 responses (limit, used, tier) | Account profiling |
| L4 | No CSP report-uri for XSS attempt visibility | Blind to attacks |
| L5 | 120s Gunicorn timeout with 4 workers (slowloris) | DoS |
| L6 | prod_checklist.yaml shows 0.0.0.0 bind (misleading docs) | Misconfiguration risk |
| L7 | TEMPORA_CLUSTER_SECRET derived from SECRET_KEY | Key compromise chain |
| L8 | sys.path.insert(0) for agents dir — module shadowing risk | Code injection if dir writable |
| L9 | No Access-Control-Max-Age on CORS preflight | Performance |
| L10 | `|safe` on whitepaper body (stored XSS if staff compromised) | XSS |

---

## Exec() Sandbox Re-enablement Criteria

Before `execute_code` can be re-enabled, it needs:
1. Container-based execution (nsjail, gVisor, or Docker with seccomp)
2. Network isolation (no outbound access from sandbox)
3. Filesystem isolation (read-only mount, no access to host)
4. Resource limits (CPU time, memory, process count)
5. No module attribute access (AST validation as defense-in-depth)
6. Enterprise-only gate at minimum

---
*Last updated: 2026-02-26*
