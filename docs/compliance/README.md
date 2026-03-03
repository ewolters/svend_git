# SOC 2 Compliance — Svend Platform

**Owner:** Eric (Founder)
**Last Updated:** 2026-03-03
**Target Audit Type:** SOC 2 Type II
**Audit Window:** TBD (minimum 3-month observation period required for Type II)

## What This Is

This directory contains all documentation, policies, checklists, and evidence pointers needed to achieve and maintain SOC 2 Type II compliance for the Svend platform (svend.ai).

## Trust Services Criteria — Scope

All five TSC categories are in scope:

| Criteria | Why |
|---|---|
| **Security** (CC1-CC9) | Required for all SOC 2 engagements |
| **Availability** (A1) | SaaS platform — customers depend on uptime |
| **Processing Integrity** (PI1) | Statistical analysis platform — computation accuracy is the product |
| **Confidentiality** (C1) | Customer analysis data, project hypotheses, proprietary datasets |
| **Privacy** (P1) | PII collected: email, name, Stripe customer IDs, chat messages |

## Platform Overview

- **Stack:** Django 5.x, PostgreSQL, Gunicorn, Caddy, Cloudflare Tunnel
- **Hosting:** Single dedicated server (Ubuntu 22.04)
- **Billing:** Stripe (Free / Professional $49 / Team $99 / Enterprise $299)
- **Encryption:** Fernet field-level (at-rest), AES-256-CBC (backups), TLS 1.2+ (transit)
- **Auth:** Django sessions, PBKDF2 password hashing, decorator-based RBAC
- **Third Parties:** Stripe, Anthropic (Claude API), Cloudflare, SMTP provider

## Current Status

**Phase: Documentation + Gap Remediation (Pre-Audit)**

See [gap-analysis.md](gap-analysis.md) for current state assessment and [checklists/remediation.md](checklists/remediation.md) for the active fix tracker.

## Directory Structure

```
docs/compliance/
├── README.md                  # This file
├── soc2-controls.md           # Control matrix mapped to TSC criteria
├── gap-analysis.md            # Current state vs. required state
├── policies/                  # Formal policy documents
│   ├── information-security.md
│   ├── access-control.md
│   ├── change-management.md
│   ├── incident-response.md
│   ├── risk-assessment.md
│   ├── vendor-management.md
│   ├── data-classification.md
│   ├── bcdr.md
│   ├── acceptable-use.md
│   └── encryption.md
├── checklists/
│   ├── pre-audit.md           # Auditor readiness checklist
│   ├── evidence-collection.md # Evidence inventory + locations
│   └── remediation.md         # Gap-to-fix tracker
└── evidence/
    └── README.md              # Evidence artifact organization guide
```

## How to Use

1. **Start with** [gap-analysis.md](gap-analysis.md) to understand where we stand
2. **Track progress** via [checklists/remediation.md](checklists/remediation.md)
3. **Before audit**, run through [checklists/pre-audit.md](checklists/pre-audit.md)
4. **Gather evidence** using [checklists/evidence-collection.md](checklists/evidence-collection.md)
5. **Policies** in `policies/` are the documents the auditor reviews directly

## Review Cadence

- Policies: Annual review (or after significant infrastructure changes)
- Gap analysis: Update after each remediation sprint
- Control matrix: Update when controls change or new criteria added
- Remediation tracker: Living document, updated continuously
