# Information Security Policy

**Policy ID:** ISP-001
**Version:** 1.0
**Effective Date:** 2026-03-03
**Owner:** Eric (Founder)
**Review Cycle:** Annual (next: 2027-03-03)

---

## 1. Purpose

Establish the overarching framework for protecting the confidentiality, integrity, and availability of Svend platform systems and customer data. This policy is the root document; all sub-policies operate under its authority.

## 2. Scope

This policy applies to:
- All Svend platform infrastructure (servers, databases, network, cloud services)
- All personnel with access to Svend systems (currently: founder + AI collaborator)
- All customer data processed, stored, or transmitted by Svend
- All third-party services integrated with Svend

## 3. Policy Statement

Svend is committed to:
- Protecting customer data with encryption at rest and in transit
- Maintaining least-privilege access to all systems
- Detecting, responding to, and recovering from security incidents
- Complying with SOC 2 Trust Services Criteria
- Continuously improving security posture through regular assessment

## 4. Information Security Principles

1. **Defense in depth** — Multiple layers: Cloudflare WAF, Caddy reverse proxy, Django application controls, database encryption
2. **Least privilege** — Access granted only as needed; tier-based feature gating; org-level role restrictions
3. **Encryption by default** — TLS 1.2+ in transit; Fernet at rest for PII; AES-256 for backups
4. **Fail secure** — Authentication failures deny access; encryption failures raise exceptions (not fallback to plaintext)
5. **Accountability** — All changes logged; security events tracked; audit trail maintained

## 5. Sub-Policies

This policy is implemented through the following specific policies:

| Policy | Document | Scope |
|---|---|---|
| Access Control | [access-control.md](access-control.md) | Authentication, authorization, session management |
| Change Management | [change-management.md](change-management.md) | Code changes, deployments, infrastructure updates |
| Incident Response | [incident-response.md](incident-response.md) | Detection, triage, containment, recovery |
| Risk Assessment | [risk-assessment.md](risk-assessment.md) | Risk identification, scoring, treatment |
| Vendor Management | [vendor-management.md](vendor-management.md) | Third-party assessment and monitoring |
| Data Classification | [data-classification.md](data-classification.md) | Data categorization and handling rules |
| Business Continuity | [bcdr.md](bcdr.md) | Backup, disaster recovery, availability |
| Acceptable Use | [acceptable-use.md](acceptable-use.md) | System usage obligations |
| Encryption | [encryption.md](encryption.md) | Cryptographic standards and key management |

## 6. Roles and Responsibilities

| Role | Responsibility |
|---|---|
| **Founder/System Owner** (Eric) | Overall security accountability; policy approval; incident escalation point; system administration |
| **AI Collaborator** (Claude) | Code review; security analysis; change implementation; documentation |
| **Third-Party Services** | Compliance with their own security certifications (Stripe: PCI DSS; Cloudflare: SOC 2; Anthropic: SOC 2) |

## 7. Compliance

- SOC 2 Type II (target)
- GDPR awareness (EU customers)
- PCI DSS compliance delegated to Stripe (no card data touches Svend servers)

## 8. Enforcement

Violations of this policy or sub-policies will be addressed through:
- Immediate remediation of the security issue
- Root cause analysis and documentation
- Policy update if the violation reveals a gap
- For third-party violations: vendor review and potential termination

## 9. Exceptions

Exceptions must be:
- Documented with justification
- Time-limited (maximum 90 days)
- Approved by the system owner
- Tracked in `.kjerne/DEBT.md` with appropriate priority

## 10. Review and Updates

- Annual review of all policies
- Immediate review after any security incident
- Update required when infrastructure significantly changes
- All reviews documented with date and findings
