# Incident Response Plan

**Policy ID:** IRP-001
**Version:** 1.0
**Effective Date:** 2026-03-03
**Last Updated:** 2026-03-05
**Owner:** Eric (Founder)
**Review Cycle:** Annual + after every incident
**Parent Policy:** [Information Security Policy](information-security.md)

---

## 1. Purpose

Define how Svend detects, responds to, contains, eradicates, and recovers from security incidents and service disruptions.

## 2. Scope

- Security incidents (unauthorized access, data breach, malware, credential compromise)
- Service disruptions (outages, degraded performance, data corruption)
- Third-party incidents (vendor breach, API outage)
- Data loss events

## 3. Severity Classification

| Severity | Definition | Response Time | Examples |
|---|---|---|---|
| **SEV-1 (Critical)** | Active data breach; complete service outage; credential compromise | Immediate (< 1 hour) | Customer data exfiltrated; database corruption; admin account compromised |
| **SEV-2 (High)** | Potential data exposure; partial outage; security vulnerability actively exploited | < 4 hours | Exploitable vulnerability discovered in production; single-service failure |
| **SEV-3 (Medium)** | Security anomaly; degraded performance; failed security control | < 24 hours | Unusual login patterns; backup failure; SSL certificate nearing expiry |
| **SEV-4 (Low)** | Minor security event; informational | < 72 hours | Failed login attempts below threshold; dependency vulnerability (unexploitable) |

## 4. Detection

### 4.1 Current Detection Capabilities

| Source | What It Detects | Location |
|---|---|---|
| fail2ban | SSH brute-force attempts | `fail2ban-svend.conf` |
| Cloudflare | DDoS attacks, bot traffic, WAF rule triggers | Cloudflare dashboard |
| Gunicorn logs | HTTP errors, timeout patterns | `/var/log/svend/access.log` |
| Django security log | CSRF failures, suspicious requests | `logs/security.log` |
| Caddy logs | Request patterns, TLS errors | `/var/log/caddy/svend.log` |
| Stripe webhooks | Payment fraud, subscription anomalies | `billing.py` |

### 4.2 Partial Detection (Implemented 2026-03-05)

| Source | What It Detects | Location |
|---|---|---|
| LoginAttempt model | Failed login surges per username; account lockout triggers (5 failures → 15-min lockout) | `accounts/models.py` |
| DRF throttling | Per-IP request rate anomalies | `settings.py` REST_FRAMEWORK |

### 4.3 Detection Gaps (Planned)

- Application-level anomaly detection (unusual data access patterns)
- Real-time alerting on error rate spikes
- Uptime monitoring with external health checks
- Failed login surge **alerting** (data is captured via LoginAttempt, but no active alerting yet)

## 5. Response Procedure

### Phase 1: Triage (0-15 minutes)

1. **Confirm the incident** — verify it's real, not a false positive
2. **Classify severity** using the table above
3. **Assign incident ID** — format: `INC-YYYY-MM-DD-NNN`
4. **Begin incident log** — timestamped notes of all actions taken

### Phase 2: Containment (15 minutes - 2 hours)

**Immediate containment (stop the bleeding):**

| Incident Type | Containment Action |
|---|---|
| Credential compromise | Rotate affected credentials; invalidate sessions; change SSH keys |
| Active exploitation | Block attacker IP via Cloudflare/UFW; disable affected endpoint |
| Data breach | Isolate affected data; revoke access tokens |
| Service outage | Identify failing component; restart service; failover if available |
| Malware | Isolate affected system; preserve forensic evidence |

**Do NOT:**
- Delete logs or evidence
- Communicate externally before understanding scope
- Make unrelated changes during incident

### Phase 3: Eradication (2-24 hours)

1. **Root cause analysis** — determine how the incident occurred
2. **Remove threat** — patch vulnerability, remove malware, close access vector
3. **Verify eradication** — confirm the threat is fully removed
4. **Harden** — implement additional controls to prevent recurrence

### Phase 4: Recovery (24-72 hours)

1. **Restore service** — from backups if needed (see [bcdr.md](bcdr.md))
2. **Verify integrity** — confirm data integrity post-recovery
3. **Monitor closely** — watch for recurrence in the following 72 hours
4. **Re-enable** — gradually restore full functionality

### Phase 5: Post-Incident (within 1 week)

1. **Post-mortem document:**
   - Timeline of events
   - Root cause
   - What worked, what didn't
   - Action items with owners and deadlines
2. **Update DEBT.md** with any systemic issues discovered
3. **Update this plan** if the incident revealed gaps
4. **Customer notification** if data was exposed (see Communication Plan below)

## 6. Communication Plan

### Internal

| Audience | Channel | Timing |
|---|---|---|
| System owner (Eric) | Direct (incident detected personally or via alerts) | Immediate |
| AI collaborator | Session conversation | During response |

### External

| Audience | Channel | Timing | Template |
|---|---|---|---|
| Affected customers | Email | Within 72 hours of confirmed breach | Breach notification template (below) |
| All customers | Status page / email | During extended outage (> 1 hour) | Service disruption template |
| Regulators | As required by jurisdiction | Per legal requirements | Legal review first |

### Breach Notification Template

```
Subject: Security Notice — Svend Platform

We are writing to inform you of a security incident that may have affected
your account on the Svend platform.

What happened: [Brief description]
When: [Date/time range]
What data was affected: [Specific data types]
What we've done: [Containment and remediation actions]
What you should do: [User actions — e.g., change password, review activity]
Contact: [Email for questions]

We take the security of your data seriously and are committed to transparency
about this incident.
```

## 7. Incident Log Template

```markdown
## INC-YYYY-MM-DD-NNN

**Severity:** SEV-X
**Status:** Active / Contained / Resolved
**Detected:** YYYY-MM-DD HH:MM UTC
**Resolved:** YYYY-MM-DD HH:MM UTC
**Duration:** Xh Ym

### Timeline
- HH:MM — [Event/action]
- HH:MM — [Event/action]

### Root Cause
[Description]

### Impact
- Users affected: [count]
- Data affected: [description]
- Service impact: [description]

### Remediation
- [Action taken]
- [Action taken]

### Action Items
- [ ] [Follow-up action] — Owner: [name] — Due: [date]
```

## 8. Testing

- **Tabletop exercise:** Annual walkthrough of a simulated SEV-1 incident
- **Recovery test:** Annual backup restoration test (see [bcdr.md](bcdr.md))
- **Post-incident review:** After every actual incident

## 9. Contacts

| Role | Name | Contact |
|---|---|---|
| System Owner / Incident Commander | Eric | [primary contact] |
| Cloudflare Support | Cloudflare | Dashboard / support ticket |
| Stripe Support | Stripe | Dashboard / support@stripe.com |
| Anthropic Support | Anthropic | Dashboard / support channel |

<!-- policy-watches: accounts/models.py:LoginAttempt -->
