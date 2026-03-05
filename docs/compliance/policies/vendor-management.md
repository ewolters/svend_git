# Vendor Management Policy

**Policy ID:** VMP-001
**Version:** 1.0
**Effective Date:** 2026-03-03
**Owner:** Eric (Founder)
**Review Cycle:** Annual
**Parent Policy:** [Information Security Policy](information-security.md)

---

## 1. Purpose

Define how third-party vendors and service providers are assessed, onboarded, monitored, and offboarded to manage supply chain risk.

## 2. Scope

All third-party services that:
- Process, store, or transmit Svend customer data
- Provide infrastructure services to the platform
- Have access to Svend systems or credentials

## 3. Vendor Inventory

### Critical Vendors (Data Processors)

| Vendor | Service | Data Exposure | Compliance | Contract |
|---|---|---|---|---|
| **Stripe** | Payment processing | Customer email, subscription tier, payment method (card via Stripe.js — never touches Svend servers) | PCI DSS Level 1, SOC 2 Type II | Stripe Services Agreement |
| **Anthropic** | LLM API (Claude) | User queries, analysis context (enterprise tier only) | SOC 2 Type II | API Terms of Service |
| **Cloudflare** | CDN, WAF, DDoS protection, Tunnel | All HTTP traffic (encrypted in transit) | SOC 2 Type II, ISO 27001 | Cloudflare Service Agreement |

### Infrastructure Vendors

| Vendor | Service | Data Exposure | Compliance |
|---|---|---|---|
| **SMTP Provider** | Transactional email | Recipient email addresses, email content | Varies by provider |
| **Let's Encrypt** (via Caddy) | TLS certificates | Domain names only | N/A (free, automated) |
| **GitHub** | Source code hosting | Source code, git history | SOC 2 Type II |

### Development Tools

| Vendor | Service | Data Exposure | Notes |
|---|---|---|---|
| **Anthropic** (Claude Code) | AI development assistant | Source code during development sessions | No persistent storage of code by Anthropic |

## 4. Vendor Assessment Criteria

### 4.1 Pre-Onboarding Assessment

Before integrating a new vendor, evaluate:

| Criterion | Weight | Assessment Method |
|---|---|---|
| Security certifications (SOC 2, ISO 27001, PCI DSS) | High | Request compliance reports |
| Data processing practices | High | Review DPA / privacy policy |
| Encryption (transit + at-rest) | High | Technical documentation |
| Incident response capability | Medium | SLA review |
| Business continuity | Medium | Uptime SLA, disaster recovery documentation |
| Sub-processor chain | Medium | Data processing agreement |
| Jurisdiction / data residency | Medium | Legal review |

### 4.2 Risk Classification

| Risk Level | Criteria | Review Frequency |
|---|---|---|
| **Critical** | Processes customer PII or financial data | Every 6 months |
| **High** | Has access to infrastructure or source code | Annual |
| **Medium** | Provides supporting services (email, monitoring) | Annual |
| **Low** | No data access; development tools only | Biennial |

## 5. Ongoing Monitoring

### 5.1 Review Checklist (Annual)

For each critical/high vendor:

- [ ] Current SOC 2 / compliance report obtained
- [ ] Any security incidents disclosed in the past year?
- [ ] Data processing agreement still current?
- [ ] Sub-processor list unchanged?
- [ ] SLA performance reviewed (uptime, response time)
- [ ] Any material changes to their security posture?
- [ ] Our integration still follows least-privilege principle?

### 5.2 Continuous Monitoring

| Signal | Source | Action |
|---|---|---|
| Vendor security breach announcement | News, vendor communications | Assess impact on Svend; rotate credentials if affected |
| Service degradation | Application error logs, uptime monitoring | Contact vendor; activate fallback if available |
| Compliance certification lapse | Annual review | Escalate; assess continued use |
| Pricing / terms change | Vendor communication | Review impact; assess alternatives |

## 6. Vendor Access Controls

| Principle | Implementation |
|---|---|
| Least privilege | Vendors receive only the access needed for their function |
| Credential isolation | Each vendor has separate API keys; no shared credentials |
| Key rotation | API keys rotatable; rotate on suspected compromise |
| No persistent access | No vendor has standing SSH or database access |
| Webhook verification | Stripe webhooks verified via HMAC signature |

## 7. Offboarding

When terminating a vendor relationship:

1. Revoke all API keys and access tokens
2. Remove vendor from any firewall allowlists
3. Verify no residual data at vendor (request deletion confirmation)
4. Update this inventory
5. Update application code to remove integration
6. Log change in `log.md`

## 8. Current Vendor Compliance Status

| Vendor | Last SOC 2 Report | Status |
|---|---|---|
| Stripe | Available via Stripe Trust Center | Verified |
| Anthropic | Available on request | To be obtained |
| Cloudflare | Available via Cloudflare Trust Hub | Verified |
| SMTP Provider | TBD | To be assessed |
| GitHub | Available via GitHub Security page | Verified |
