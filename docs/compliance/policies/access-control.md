# Access Control Policy

**Policy ID:** ACP-001
**Version:** 1.0
**Effective Date:** 2026-03-03
**Owner:** Eric (Founder)
**Review Cycle:** Annual
**Parent Policy:** [Information Security Policy](information-security.md)

---

## 1. Purpose

Define controls for authenticating users, authorizing access to resources, and managing sessions across the Svend platform.

## 2. Scope

- Application-level user authentication and authorization
- Infrastructure access (SSH, database, server administration)
- Third-party service access (Stripe, Anthropic, Cloudflare)
- API access and rate limiting

## 3. Authentication Controls

### 3.1 Password Requirements

| Control | Implementation | Reference |
|---|---|---|
| Minimum length | Django MinimumLengthValidator (8 chars) | `settings.py` AUTH_PASSWORD_VALIDATORS |
| Complexity | NumericPasswordValidator + CommonPasswordValidator | `settings.py` |
| Similarity check | UserAttributeSimilarityValidator | `settings.py` |
| Hashing algorithm | PBKDF2-SHA256 (migration to Argon2id planned) | Django PASSWORD_HASHERS |

### 3.2 Multi-Factor Authentication

| Status | Plan |
|---|---|
| **Not yet implemented** | TOTP-based MFA via pyotp — Phase 2 remediation |
| | Mandatory for enterprise tier accounts |
| | Optional (encouraged) for all other tiers |

### 3.3 Session Management

| Control | Current State | Target |
|---|---|---|
| Session duration | 14 days (Django default) | 8 hours absolute, 1 hour idle |
| Cookie security | HTTPOnly, Secure, SameSite=Lax | No change needed |
| Session invalidation on password change | Not implemented | Invalidate all sessions on password reset |
| Concurrent session limit | No limit | Monitor but don't limit (user experience) |

### 3.4 Account Lockout

| Control | Current State | Target |
|---|---|---|
| Failed login lockout | None | Lock for 15 minutes after 5 failed attempts |
| Notification | None | Email notification on lockout |
| Username enumeration | Distinct error messages | Generic "invalid credentials" response |

### 3.5 Email Verification

| Control | Implementation |
|---|---|
| Required for account activation | Yes — token sent via email |
| Token storage | SHA-256 hashed |
| Token expiry | None (gap — target: 48 hours) |

## 4. Authorization Model

### 4.1 Application-Level RBAC

Access controlled via decorator-based permission system (`accounts/permissions.py`):

| Decorator | Access Level | Description |
|---|---|---|
| `@require_auth` | Any authenticated user | Basic authentication gate |
| `@rate_limited` | Authenticated + rate cap | Per-tier daily query limits |
| `@require_paid` | Founder/Pro/Team/Enterprise | Paid feature gate |
| `@require_team` | Team/Enterprise | Collaboration features |
| `@require_enterprise` | Enterprise only | AI/LLM features (Claude access) |
| `@require_org_admin` | Org admin/owner | Organization management |
| `@gated_paid` | Paid + feature flag | Full tools access + rate limiting |
| `@allow_guest` | Guest token bearer | Whiteboard guest access (scoped) |

### 4.2 Multi-Tenancy Roles

Within organizations (`core/models/tenant.py`):

| Role | Permissions |
|---|---|
| Owner | Full access; manage members; billing; delete org |
| Admin | Manage members; manage projects; all features |
| Member | Use features; create/edit own resources |
| Viewer | Read-only access to shared resources |

### 4.3 Resource Ownership

- All API endpoints validate `user=request.user` or `tenant=` membership
- Cross-tenant data access prevented by query filters
- Guest tokens scoped to specific whiteboard boards

## 5. Infrastructure Access

### 5.1 Server Access

| Control | Implementation |
|---|---|
| SSH authentication | Key-only (password auth disabled) |
| SSH brute-force protection | fail2ban: 3 attempts, 24-hour ban |
| Firewall | UFW — only SSH + HTTP(S) open |
| Direct port exposure | None — all traffic via Cloudflare Tunnel |

### 5.2 Database Access

| Control | Implementation |
|---|---|
| Network access | localhost only (PostgreSQL bound to 127.0.0.1) |
| Application access | Django ORM via environment-configured connection string |
| Direct access | psql from localhost only, as system user |

### 5.3 Third-Party Service Access

| Service | Access Control |
|---|---|
| Stripe | API key in environment variable; webhook signature verification |
| Anthropic | API key in environment variable; enterprise-tier gated |
| Cloudflare | Account credentials; tunnel token in environment |
| SMTP | Credentials in environment variables |

## 6. API Rate Limiting

| Tier | Daily Query Limit | Enforcement |
|---|---|---|
| Free | 15 | `@rate_limited` decorator |
| Founder | 250 | `@rate_limited` |
| Professional | 500 | `@rate_limited` |
| Team | 750 | `@rate_limited` |
| Enterprise | 1500 | `@rate_limited` |

Rate limit responses return HTTP 429 with remaining quota information.

## 7. Onboarding and Offboarding

### Onboarding
1. User registers with email + password
2. Email verification required before account activation
3. Tier assigned based on subscription (default: Free)
4. Org membership granted via invitation (7-day expiry)

### Offboarding
- Subscription cancellation downgrades to Free tier
- Org removal revokes tenant access immediately
- Account deletion: planned (see gap-analysis.md GAP-05)
- **Gap:** No formal complete data purge on account deletion

## 8. Review

- RBAC decorator coverage reviewed with each new endpoint
- Password policy reviewed annually
- Failed login patterns reviewed monthly (once monitoring is active)
- Org role assignments reviewed quarterly by org owners
