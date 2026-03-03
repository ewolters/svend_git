# Acceptable Use Policy

**Policy ID:** AUP-001
**Version:** 1.0
**Effective Date:** 2026-03-03
**Owner:** Eric (Founder)
**Review Cycle:** Annual
**Parent Policy:** [Information Security Policy](information-security.md)

---

## 1. Purpose

Define acceptable and prohibited uses of Svend platform systems, data, and resources for all personnel with access.

## 2. Scope

- System owner (Eric) — full system access
- AI collaborator (Claude) — development access during sessions
- Future employees or contractors — any system access
- Platform users — covered separately by Terms of Service

## 3. General Principles

- Systems and data are to be used for legitimate business purposes
- Access is granted on a least-privilege basis
- All actions on production systems should be intentional and documented
- When in doubt about whether an action is appropriate, ask before acting

## 4. Acceptable Use

### 4.1 Production Systems

| Allowed | Conditions |
|---|---|
| Application deployment | Via documented change process (see [change-management.md](change-management.md)) |
| Database access | For troubleshooting, migration, or backup purposes only |
| Log review | For debugging, security investigation, or performance analysis |
| Configuration changes | Documented in `log.md`; backed up first |
| Backup management | Via established scripts; encryption maintained |

### 4.2 Customer Data

| Allowed | Conditions |
|---|---|
| Viewing for support | Only when investigating a customer-reported issue |
| Aggregate analysis | For platform improvement (anonymized/aggregated) |
| Database queries | Read-only; never modify customer data directly in database |

### 4.3 Development

| Allowed | Conditions |
|---|---|
| Code development | Using version control; reviewed before deployment |
| Testing | On non-production data; never use real customer data for testing |
| Dependency installation | Reviewed for security; from trusted sources only |

## 5. Prohibited Use

### 5.1 Strictly Prohibited

- Accessing customer data without legitimate business reason
- Sharing credentials, API keys, or encryption keys
- Disabling security controls (firewall, fail2ban, CSRF) without documented justification
- Installing unauthorized software on the production server
- Using production data for testing or development
- Storing secrets in source code or version control
- Bypassing authentication or authorization controls
- Deliberately degrading platform availability

### 5.2 Prohibited Without Approval

- Direct database modification of customer data
- Changing firewall rules or network configuration
- Rotating encryption keys
- Granting system access to new personnel
- Installing new third-party services or integrations

## 6. Data Handling Obligations

All personnel with system access must:

1. **Protect credentials** — use unique, strong passwords; never share; use SSH keys for server access
2. **Encrypt sensitive data** — follow [data-classification.md](data-classification.md) handling rules
3. **Report incidents** — immediately report suspected security events per [incident-response.md](incident-response.md)
4. **Minimize data exposure** — don't copy customer data to personal devices; don't include PII in logs
5. **Secure workstations** — screen lock; full-disk encryption; updated OS and software

## 7. AI Collaborator Specific Guidelines

The AI collaborator (Claude Code sessions) operates under these additional constraints:

- No destructive git operations (force push, reset --hard) without explicit approval
- No direct database modifications
- No credential or key access (keys loaded from environment, not readable)
- All code changes reviewed before deployment
- Sessions do not persist — no standing access to systems

## 8. Enforcement

Violations are addressed through:
- Immediate revocation of access if active threat
- Root cause analysis
- Policy update if gap identified
- For contractors: contract termination for serious violations

## 9. Acknowledgment

All personnel with system access must acknowledge this policy. For the current single-operator setup, this document serves as the acknowledged standard of conduct.

| Name | Role | Acknowledged | Date |
|---|---|---|---|
| Eric | Founder / System Owner | Yes | 2026-03-03 |
