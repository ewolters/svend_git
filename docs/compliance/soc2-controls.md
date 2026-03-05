# SOC 2 Control Matrix — Svend Platform

**Last Updated:** 2026-03-05 (CHG-001 v1.6 lockdown + SOC 2 automation)

Status key: **Met** = control operating | **Partial** = control exists with gaps | **Gap** = control not yet implemented

---

## CC1 — Control Environment

| ID | Criterion | Control Description | Owner | Evidence Location | Status |
|---|---|---|---|---|---|
| CC1.1 | Demonstrates commitment to integrity and ethical values | Acceptable use policy; code of conduct for platform use | Eric | `policies/acceptable-use.md` | **Partial** — policy drafted, no signed acknowledgment process |
| CC1.2 | Board exercises oversight | Founder reviews all changes; AI collaborator provides second review on all code | Eric | `log.md`, git history | **Met** — small org, founder has direct oversight |
| CC1.3 | Management establishes structure, authority, and responsibility | RBAC system with tier-based permissions; org roles (owner/admin/member/viewer) | Eric | `accounts/permissions.py`, `core/models/tenant.py` | **Met** |
| CC1.4 | Demonstrates commitment to competence | Domain expertise documented; continuous improvement methodology encoded in platform | Eric | CLAUDE.md | **Met** |
| CC1.5 | Enforces accountability | 3-layer enforcement: model `clean()` rejects bad CRs, `validate_for_transition()` blocks state changes, daily `change_management` check (14 checks) flags all gaps | Eric | `syn/audit/models.py`, `syn/audit/compliance.py`, `api/internal_views.py` | **Met** — CHG-001 v1.6 automated enforcement |

## CC2 — Communication and Information

| ID | Criterion | Control Description | Owner | Evidence Location | Status |
|---|---|---|---|---|---|
| CC2.1 | Uses relevant, quality information | Application logs (svend.log, security.log); Gunicorn access logs; Caddy request logs | Eric | `/var/log/svend/`, `logs/` | **Partial** — logs exist, no centralized monitoring/alerting |
| CC2.2 | Communicates internally | Change log, debt tracker, git commits document all decisions | Eric | `log.md`, git history | **Met** |
| CC2.3 | Communicates externally | Terms of service, privacy policy on svend.ai; Stripe handles billing communications | Eric | `site/index.html`, Stripe dashboard | **Partial** — ToS/privacy policy need SOC 2 alignment review |

## CC3 — Risk Assessment

| ID | Criterion | Control Description | Owner | Evidence Location | Status |
|---|---|---|---|---|---|
| CC3.1 | Specifies suitable objectives | Platform roadmap and architecture documented; TSC scope defined | Eric | `ARCHITECTURE.md`, this folder | **Met** |
| CC3.2 | Identifies and analyzes risk | Security debt tracked with priority levels (P0-P3); gap analysis performed | Eric | `.kjerne/DEBT.md`, `gap-analysis.md` | **Partial** — needs formal risk register with likelihood/impact scoring |
| CC3.3 | Considers potential for fraud | Input validation on all API endpoints; CSRF protection; rate limiting per tier; Synara DSL parser rejects injection-like inputs; LLM prompt injection hardened with structured message separation | Eric | Django views, `permissions.py`, `synara/dsl.py`, `synara/llm_interface.py` | **Partial** — no formal fraud risk assessment |
| CC3.4 | Identifies and assesses changes | ChangeRequest model with lifecycle tracking; daily `change_management` compliance check detects gaps automatically; `validate_for_transition()` blocks bad state transitions; multi-agent risk assessment for features/migrations | Eric | `syn/audit/models.py`, `syn/audit/compliance.py` | **Met** — CHG-001 v1.6 automated change detection |

## CC4 — Monitoring Activities

| ID | Criterion | Control Description | Owner | Evidence Location | Status |
|---|---|---|---|---|---|
| CC4.1 | Selects and develops monitoring | Application logs with rotation; Gunicorn access/error logs; Caddy JSON logs | Eric | Log configs in `settings.py`, `gunicorn.conf.py`, `Caddyfile` | **Partial** — logging exists, no active monitoring/alerting system |
| CC4.2 | Evaluates and communicates deficiencies | Debt tracker with weekly review; security items prioritized P0-P1 | Eric | `.kjerne/DEBT.md` | **Met** |

## CC5 — Control Activities

| ID | Criterion | Control Description | Owner | Evidence Location | Status |
|---|---|---|---|---|---|
| CC5.1 | Selects and develops control activities | RBAC decorators; input validation; CSRF; rate limiting; encryption at rest/transit | Eric | `permissions.py`, `settings.py`, `encryption.py` | **Met** |
| CC5.2 | Selects and develops technology controls | Cloudflare WAF/DDoS; fail2ban (SSH: 3 retries/24hr ban); UFW firewall; systemd resource limits | Eric | `Caddyfile`, `fail2ban-svend.conf`, systemd units | **Met** |
| CC5.3 | Deploys through policies and procedures | Policies documented in this folder; deployment via git pull + service restart | Eric | `policies/`, `start_prod.sh` | **Partial** — no automated deployment pipeline |

## CC6 — Logical and Physical Access

| ID | Criterion | Control Description | Owner | Evidence Location | Status |
|---|---|---|---|---|---|
| CC6.1 | Implements logical access security | SSH key-only access; fail2ban; application behind Cloudflare Tunnel (no direct port exposure) | Eric | `fail2ban-svend.conf`, Cloudflare config | **Met** |
| CC6.2 | Authenticates users prior to access | Email/password authentication; email verification required; session-based auth | Eric | `accounts/models.py`, `accounts/views.py` | **Partial** — no MFA |
| CC6.3 | Manages access to infrastructure | Single admin (Eric); SSH key-only; systemd service isolation | Eric | Server config | **Met** — single operator, full control |
| CC6.4 | Restricts logical access to software | Tier-based feature gating; org-level role checks; per-resource ownership validation; IDOR audit completed across Synara endpoints (2026-03) | Eric | `permissions.py`, `synara_views.py` | **Met** |
| CC6.5 | Restricts physical access | Server located in controlled environment | Eric | Physical access controls | **Met** |
| CC6.6 | Manages system access during lifecycle | User accounts deactivated on subscription end; org invites expire in 7 days | Eric | `accounts/models.py`, `tenant.py` | **Partial** — no formal offboarding/deprovisioning workflow |
| CC6.7 | Manages changes to infrastructure | ChangeRequest model with full lifecycle (draft → completed); immutable ChangeLog chain; risk assessment gating; commit SHA linkage; daily compliance monitoring | Eric | `syn/audit/models.py`, `syn/audit/compliance.py`, `log.md` | **Met** — CHG-001 v1.6 automated change management |
| CC6.8 | Manages security vulnerabilities | Security issues tracked as P0/P1 in DEBT.md; prompt injection hardened (Synara LLM interface, DSL parser); IDOR audit completed (Synara migration 2026-03); cache bounds enforced | Eric | `.kjerne/DEBT.md`, `log.md`, `synara/` | **Partial** — no automated vulnerability scanning |

## CC7 — System Operations

| ID | Criterion | Control Description | Owner | Evidence Location | Status |
|---|---|---|---|---|---|
| CC7.1 | Detects and monitors for vulnerabilities | Manual security review; DEBT.md prioritization; dependency review | Eric | `.kjerne/DEBT.md` | **Gap** — no automated scanning (Dependabot, Snyk, etc.) |
| CC7.2 | Monitors for anomalies | Gunicorn/Caddy logs; fail2ban for SSH; Cloudflare analytics | Eric | Log files, Cloudflare dashboard | **Partial** — no application-level anomaly detection |
| CC7.3 | Evaluates security events | Manual log review; debt tracker for identified issues | Eric | `.kjerne/DEBT.md` | **Partial** — no SIEM or automated triage |
| CC7.4 | Implements incident response | Incident response plan documented | Eric | `policies/incident-response.md` | **Partial** — policy drafted, not yet tested via tabletop exercise |
| CC7.5 | Identifies and addresses system faults | Gunicorn auto-restart on failure; systemd service monitoring; worker recycling every 1000 requests | Eric | `gunicorn.conf.py`, systemd units | **Met** |

## CC8 — Change Management

| ID | Criterion | Control Description | Owner | Evidence Location | Status |
|---|---|---|---|---|---|
| CC8.1 | Manages changes to infrastructure and software | Git version control; manual change log; debt tracking; manual deployment | Eric | `log.md`, git history, `.kjerne/DEBT.md` | **Partial** — no CI/CD, no automated testing, no staging environment |

## CC9 — Risk Mitigation

| ID | Criterion | Control Description | Owner | Evidence Location | Status |
|---|---|---|---|---|---|
| CC9.1 | Identifies and assesses vendor risk | Vendor inventory maintained; Stripe PCI DSS compliant; Cloudflare SOC 2 certified | Eric | `policies/vendor-management.md` | **Partial** — no formal vendor assessment process |
| CC9.2 | Manages vendor relationships | Stripe webhook signature verification; Anthropic API key rotation capability; Cloudflare access controls | Eric | `billing.py`, `settings.py` | **Partial** — no regular vendor review cadence |

## A1 — Availability

| ID | Criterion | Control Description | Owner | Evidence Location | Status |
|---|---|---|---|---|---|
| A1.1 | Manages capacity | systemd memory limit (4GB); CPU quota (200%); Gunicorn worker auto-scaling; request timeout (120s); Synara graph cache bounded (max 1000 entries, 30-min TTL) | Eric | systemd units, `gunicorn.conf.py`, `synara/` | **Partial** — no auto-scaling, no capacity monitoring/alerting |
| A1.2 | Manages environmental threats | Cloudflare DDoS protection; fail2ban; UFW firewall | Eric | Cloudflare config, `fail2ban-svend.conf` | **Met** |
| A1.3 | Manages recovery operations | Daily encrypted backups (AES-256); 30-day retention; documented restore procedure | Eric | `backup_db.sh`, systemd timer | **Partial** — backups on same machine, no off-site replication, no tested recovery |

## PI1 — Processing Integrity

| ID | Criterion | Control Description | Owner | Evidence Location | Status |
|---|---|---|---|---|---|
| PI1.1 | Uses quality information | Input validation on all API endpoints; type coercion in triage; data profiling in DSW; Synara DSL parser validates syntax before evaluation (no eval/exec) | Eric | View functions, `triage_views.py`, `synara/dsl.py` | **Met** |
| PI1.2 | System processes data accurately | 200+ statistical analyses with scipy/statsmodels; Plotly visualization; results reproducible via random seed | Eric | `dsw/`, `spc.py` | **Met** |
| PI1.3 | System processes data completely | Transaction-based processing; Django ORM atomic operations; Synara belief graph uses transaction-scoped saves with FK cascade (no orphaned data on partial failure) | Eric | Django views, `synara_views.py` | **Partial** — no end-to-end data integrity checksums |
| PI1.4 | System outputs are complete and accurate | JSON response format; Plotly traces with full statistical output; confidence intervals reported | Eric | DSW views | **Met** |
| PI1.5 | Handles input/processing errors | HTTP status codes; structured error responses; rate limit feedback | Eric | View functions | **Met** |

## C1 — Confidentiality

| ID | Criterion | Control Description | Owner | Evidence Location | Status |
|---|---|---|---|---|---|
| C1.1 | Identifies confidential information | Data classification policy defines levels; encrypted fields for PII/financial data | Eric | `policies/data-classification.md`, `encryption.py` | **Partial** — classification policy drafted, not yet enforced systematically |
| C1.2 | Protects confidential information | Fernet field-level encryption; HTTPS/HSTS; session cookies HTTPOnly + Secure + SameSite | Eric | `encryption.py`, `settings.py` | **Met** |
| C1.3 | Disposes of confidential information | Backup rotation (30 days); purge script for data cleanup | Eric | `backup_db.sh`, `run_purge.sh` | **Partial** — no formal data disposal/retention SLA |

## P1 — Privacy

| ID | Criterion | Control Description | Owner | Evidence Location | Status |
|---|---|---|---|---|---|
| P1.1 | Provides notice about privacy practices | Privacy policy on svend.ai | Eric | `site/index.html` | **Partial** — needs SOC 2 alignment review |
| P1.2 | Provides choice and consent | Email verification required; optional profile fields; cookie consent | Eric | Registration flow | **Partial** — no granular consent management |
| P1.3 | Collects PII for identified purposes | PII limited to auth (email, name), billing (Stripe ID encrypted), and optional profile fields | Eric | `accounts/models.py` | **Met** |
| P1.4 | Limits PII usage | PII used only for auth, billing, and user-requested features; analytics uses hashed IPs | Eric | `api/models.py` (SiteVisit) | **Met** |
| P1.5 | Retains PII as needed | Purge script; backup rotation; email tokens cleared on verification | Eric | `run_purge.sh` | **Partial** — no formal retention schedule |
| P1.6 | Disposes of PII | Account deletion capability; backup rotation | Eric | Account management | **Partial** — no verified complete data deletion (all tables) |
| P1.7 | Provides quality PII | Users can update their own profile; email change with re-verification | Eric | Account settings | **Met** |
| P1.8 | Provides access and correction | Users access own data via application; no formal data export (GDPR-style) | Eric | Application UI | **Gap** — no self-service data export |
