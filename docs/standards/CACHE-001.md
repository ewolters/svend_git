# CACHE-001: Caching Patterns & HTTP Cache Control

**Version:** 1.0
**Status:** APPROVED
**Date:** 2026-03-04
**Author:** Eric + Claude (Systems Architect)

**Compliance:**
- DOC-001 ≥ 1.2 (Documentation Structure — §7 Machine-Readable Hooks)
- XRF-001 ≥ 1.0 (Cross-Reference Syntax)
- MAP-001 ≥ 1.0 (Architecture Mapping)
- STY-001 ≥ 1.0 (Code Style & Conventions)
- SOC 2 CC6.1 (Logical Access Security)
- SOC 2 CC7.2 (System Monitoring)
- OWASP Cache Poisoning Prevention

**Related Standards:**
- SEC-001 ≥ 1.0 (Security Architecture)
- API-001 ≥ 1.0 (API Design)
- OPS-001 ≥ 1.0 (Operations & Deployment)
- FE-001 ≥ 1.0 (Frontend Patterns)

---

## **1. SCOPE AND PURPOSE**

### 1.1 Purpose

Define caching policies across all layers of the Svend platform: HTTP response headers, application-level caches, in-memory LRU caches, browser storage, CDN resource integrity, and cache invalidation patterns. This standard prevents stale data bugs, cache poisoning, and uncontrolled memory growth.

### 1.2 Scope

- HTTP cache headers on all responses (dynamic and static)
- Django middleware cache control (`NoCacheDynamicMiddleware`, `WhiteNoise`)
- Application caches (idempotency, rate limiting, session cache)
- In-memory caches in view modules (DSW, SPC, Synara, interview)
- Browser storage (`localStorage`, `sessionStorage`)
- CDN-loaded third-party resources
- Caddy/Cloudflare cache interaction

Does NOT apply to: database query optimization, ORM `.select_related()`/`.prefetch_related()` (performance, not caching policy), Cloudflare dashboard settings (managed externally).

---

## **2. NORMATIVE REFERENCES**

| Standard | Relevance |
|----------|-----------|
| SEC-001 | CSP directives governing CDN resource loading |
| API-001 | Response envelope, idempotency middleware |
| OPS-001 | Caddy configuration, static file serving |
| FE-001 | Browser storage patterns, CDN library loading |
| ARCH-001 | Middleware ordering in settings.py |

---

## **3. TERMINOLOGY**

| Term | Definition |
|------|-----------|
| **Dynamic response** | Any Django-rendered HTML page or JSON API response |
| **Static asset** | Files served from `staticfiles/` via WhiteNoise or Caddy |
| **SRI** | Subresource Integrity — cryptographic hash verifying CDN resource content |
| **LRU cache** | Least Recently Used eviction policy for bounded in-memory storage |
| **Cache-bust** | Technique to force fresh content fetch (query param, content hash) |
| **Content-hash versioning** | WhiteNoise manifest mapping filenames to hashed variants |

---

## **4. HTTP CACHE CONTROL**

<!-- assert: NoCacheDynamicMiddleware is in MIDDLEWARE | check=cache-middleware -->
<!-- impl: syn/audit/compliance.py -->
<!-- test: syn.audit.tests.test_caching.CacheMiddlewareTest.test_no_cache_middleware_present -->
<!-- test: syn.audit.tests.test_caching.CacheMiddlewareTest.test_middleware_position -->

### 4.1 Dynamic Responses

All dynamic responses (HTML pages, API JSON) MUST include:

```
Cache-Control: no-cache, no-store, must-revalidate, private
```

Enforced by `accounts.middleware.NoCacheDynamicMiddleware`, which sets this header on any response that does not already have `Cache-Control`.

**Middleware position:** After `WhiteNoiseMiddleware` (so static files are served with their own caching headers) and before application middleware.

### 4.2 Static Assets

<!-- assert: WhiteNoise uses CompressedManifestStaticFilesStorage | check=cache-whitenoise -->
<!-- impl: syn/audit/compliance.py -->
<!-- test: syn.audit.tests.test_caching.StaticCacheTest.test_whitenoise_storage_backend -->
<!-- test: syn.audit.tests.test_caching.StaticCacheTest.test_whitenoise_in_middleware -->

Static files served by WhiteNoise use `CompressedManifestStaticFilesStorage`:

- Content-hash versioning (e.g., `app.3aa6808b4dad.js`)
- Brotli + gzip compression variants auto-generated
- Effectively infinite cache TTL (hash changes when content changes)

### 4.3 Internal Dashboard

The internal dashboard and its API endpoints MUST use `@never_cache` or equivalent cache-busting to prevent stale operational data.

### 4.4 Caddy Static File Headers

Caddy serves `/static/*` directly via `file_server`. Content-hashed files SHOULD have long-lived cache headers. The CSP header MUST be set in Django middleware only — not duplicated in Caddy.

---

## **5. APPLICATION-LEVEL CACHING**

<!-- assert: Idempotency cache has bounded TTL | check=cache-idempotency-ttl -->
<!-- impl: syn/audit/compliance.py -->
<!-- test: syn.audit.tests.test_caching.AppCacheTest.test_idempotency_ttl_bounded -->

<!-- assert: Rate limit cache uses Django cache framework with bounded TTL | check=cache-rate-limit-ttl -->
<!-- impl: syn/synara/middleware.py -->
<!-- test: syn.audit.tests.test_caching.AppCacheTest.test_idempotency_ttl_value -->

### 5.1 Idempotency Cache

| Setting | Value | Source |
|---------|-------|--------|
| TTL | 24 hours | `syn.api.middleware.IDEMPOTENCY_TTL_HOURS` |
| Backend | Django cache framework | `django.core.cache` |
| Key format | `idempotency:{tenant_id}:{key}` | — |

### 5.2 Rate Limit Cache

| Setting | Value | Source |
|---------|-------|--------|
| TTL | 3600 seconds | `syn.synara.middleware.rate_limit` |
| Backend | Django cache framework | `django.core.cache` |
| Key format | `rate_limit:{tenant_id}` | — |

### 5.3 Session Cache (Database-Backed)

<!-- assert: SessionCache uses JSON-only serialization, no pickle | check=cache-session-json -->
<!-- impl: agents_api/cache.py -->
<!-- test: syn.audit.tests.test_caching.SessionCacheTest.test_session_cache_json_only -->
<!-- test: syn.audit.tests.test_caching.SessionCacheTest.test_session_cache_has_namespaces -->

`agents_api.cache.SessionCache` — persistent, encrypted, JSON-only:

| Setting | Value |
|---------|-------|
| Serialization | JSON only (pickle rejected for security) |
| TTL | Configurable per entry |
| Namespaces | `synara` (3600s), `model` (1800s) |
| Cleanup | `cleanup_expired_cache()` via syn.sched |

---

## **6. IN-MEMORY CACHES**

<!-- assert: All in-memory caches have max size bounds | check=cache-memory-bounds -->
<!-- impl: syn/audit/compliance.py -->
<!-- test: syn.audit.tests.test_caching.MemoryCacheTest.test_dsw_cache_has_bounds -->
<!-- test: syn.audit.tests.test_caching.MemoryCacheTest.test_spc_cache_has_bounds -->
<!-- test: syn.audit.tests.test_caching.MemoryCacheTest.test_synara_cache_has_bounds -->
<!-- test: syn.audit.tests.test_caching.MemoryCacheTest.test_interview_cache_has_bounds -->

### 6.1 Requirements

All in-memory caches MUST have:
1. **Maximum size bound** — prevents unbounded memory growth
2. **LRU or TTL eviction** — oldest/expired entries removed when bound is reached
3. **Thread safety** — `threading.Lock()` or equivalent for shared state

### 6.2 Current Inventory

| Module | Variable | Max Size | TTL | Thread-Safe |
|--------|----------|----------|-----|-------------|
| `dsw_views.py` | `_model_cache` | 100 | 3600s | Yes (`Lock`) |
| `problem_views.py` | `_interview_sessions` | 128 | None | No |
| `spc_views.py` | `_parsed_data_cache` | 256 | None | No |
| `synara_views.py` | `_synara_cache` | 128 | None | No |

Known debt: interview, SPC, and Synara caches lack TTL and thread safety. Tracked in `.kjerne/DEBT.md`.

<!-- assert: @lru_cache usage has explicit maxsize set, not unbounded | check=cache-lru-maxsize -->
<!-- impl: syn/audit/compliance.py -->
<!-- test: syn.audit.tests.test_caching.LRUCacheTest.test_lru_caches_have_maxsize -->
<!-- test: syn.audit.tests.test_caching.LRUCacheTest.test_no_unbounded_lru_cache -->

### 6.3 `@lru_cache` Usage

`@lru_cache` on configuration/singleton functions is permitted when:
- `maxsize` is explicitly set (not unbounded)
- The cached value is immutable or process-lifetime
- No user-specific data is cached

Current usage: `svend_config/config.py`, `syn/core/config.py`, `agents_api/embeddings.py`, `inference/pipeline.py`.

---

## **7. BROWSER STORAGE**

<!-- assert: localStorage must not store authentication tokens, session IDs, or PII | check=cache-localstorage-no-secrets -->
<!-- impl: templates/base_app.html -->
<!-- test: syn.audit.tests.test_caching.BrowserStorageTest.test_localstorage_no_auth_tokens -->
<!-- test: syn.audit.tests.test_caching.BrowserStorageTest.test_sessionstorage_no_secrets -->

### 7.1 localStorage

| Key | Purpose | Expiry |
|-----|---------|--------|
| `svend_theme` | User theme preference | Never (user-managed) |

Rules:
- MUST NOT store authentication tokens, session IDs, or PII
- MUST NOT store data that affects server-side behavior
- Limited to UI preferences only

### 7.2 sessionStorage

| Key | Purpose | Expiry |
|-----|---------|--------|
| `verification_dismissed` | Email verification banner state | Tab close |
| `svend_sl` | Splash screen shown flag | Tab close |
| `doe_*` | DOE workflow state (step, design, analysis) | Tab close |

Rules:
- Transient UI state only
- No secrets or auth tokens
- Maximum 50KB per key (enforced by browser, documented here)

---

## **8. CDN RESOURCES**

<!-- assert: CDN script tags have version pins | check=cache-cdn-versions -->
<!-- impl: syn/audit/compliance.py -->
<!-- test: syn.audit.tests.test_caching.CDNResourceTest.test_cdn_scripts_version_pinned -->

### 8.1 Version Pinning

All CDN-loaded resources MUST use pinned versions (not `@latest`):

```html
<!-- correct -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>

<!-- prohibited -->
<script src="https://cdn.jsdelivr.net/npm/chart.js/dist/chart.umd.min.js"></script>
```

<!-- assert: CDN origins in templates are listed in CSP directives | check=cache-cdn-csp-alignment -->
<!-- impl: syn/audit/compliance.py -->
<!-- test: syn.audit.tests.test_caching.CDNResourceTest.test_cdn_scripts_version_pinned -->

### 8.2 CSP Alignment

Every CDN origin used in `<script>`, `<link>`, or `<img>` tags MUST be listed in the corresponding CSP directive in `settings.CONTENT_SECURITY_POLICY`. Cross-reference: SEC-001 §5.

### 8.3 Subresource Integrity (SRI)

CDN resources SHOULD include `integrity` and `crossorigin` attributes:

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"
        integrity="sha384-..." crossorigin="anonymous"></script>
```

Known debt: Current CDN tags lack SRI attributes. Tracked in `.kjerne/DEBT.md`.

---

## **9. CACHE INVALIDATION**

### 9.1 Static Assets

Content-hash versioning via WhiteNoise manifest provides automatic invalidation. Running `collectstatic` regenerates hashes for changed files.

### 9.2 Application Caches

| Cache | Invalidation Method |
|-------|-------------------|
| Idempotency | TTL expiry (24h) |
| Rate limits | TTL expiry (1h) |
| Session cache | TTL expiry + `clear_expired()` |
| In-memory LRU | Eviction on size bound + process restart |
| `@lru_cache` | Process restart only |

### 9.3 Browser Cache

- Dynamic pages: `Cache-Control: no-cache, no-store` prevents caching
- Dashboard API: `cache: 'no-store'` in fetch + timestamp cache-bust parameter
- Static assets: Hash-versioned URLs automatically invalidate

---

## **10. ANTI-PATTERNS**

**Prohibited:**

1. API GET endpoints without `Cache-Control` header (Cloudflare may cache)
2. In-memory caches without size bounds (memory leak risk)
3. `@lru_cache` on functions that return user-specific data
4. CDN resources loaded without version pins
5. Storing auth tokens or PII in `localStorage`
6. Duplicate CSP headers from multiple sources (Caddy + Django)
7. `pickle` serialization in any cache backend (security risk)

---

## **11. ACCEPTANCE CRITERIA**

| # | Criterion | Verified By |
|---|-----------|-------------|
| AC-1 | Dynamic responses include `Cache-Control: no-store` | `cache-check-registered` check |
| AC-2 | Static assets served with content-hash versioning | `cache-check-registered` check |
| AC-3 | In-memory caches have `maxsize` bounds | `cache-check-registered` check |
| AC-4 | CDN resources use version-pinned URLs | `cache-check-registered` check |
| AC-5 | No `localStorage` usage for auth tokens or PII | Manual review |

---

## **12. COMPLIANCE MAPPING**

<!-- assert: Caching compliance check is registered | check=cache-check-registered -->
<!-- impl: syn/audit/compliance.py -->
<!-- test: syn.audit.tests.test_caching.CheckRegistrationTest.test_check_registered -->
<!-- test: syn.audit.tests.test_caching.CheckRegistrationTest.test_check_is_callable -->
<!-- test: syn.audit.tests.test_caching.CheckRegistrationTest.test_check_returns_valid_structure -->

The `caching` compliance check validates:
- `NoCacheDynamicMiddleware` in `MIDDLEWARE` (§4)
- `WhiteNoise` uses `CompressedManifestStaticFilesStorage` (§4)
- In-memory caches have size bounds (§6)
- CDN resources use version pins (§8)

| Control | Mapping |
|---------|---------|
| SOC 2 CC6.1 | Logical Access Security — cache key isolation |
| SOC 2 CC7.2 | System Monitoring — cache invalidation patterns |
| OWASP | Cache Poisoning Prevention — no-store on dynamic |

---

## **REVISION HISTORY**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-04 | Eric + Claude | Initial release. Codifies existing cache patterns, identifies SRI and TTL debt |
