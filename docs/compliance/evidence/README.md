# Evidence Directory

**Purpose:** Store evidence artifacts collected for SOC 2 audit.

## Rules

- **No secrets.** Never store API keys, passwords, encryption keys, or database credentials here.
- **Redact PII.** Replace customer emails, names, and IPs with `[REDACTED]` in all evidence files.
- **Date everything.** Use `YYYY-MM-DD` prefix on all filenames.
- **Collect fresh.** Evidence should be collected within 30 days of audit, not stale.

## Directory Structure

```
evidence/
├── README.md              # This file
├── governance/            # Policy approvals, review records
├── access-control/        # Password config, RBAC proof, user lists
├── encryption/            # TLS certs, encryption config, key permissions
├── logging/               # Log samples (7-day extracts, redacted)
├── change-management/     # Git logs, change log excerpts, backup proof
├── incidents/             # Incident records (if any occurred)
├── bcdr/                  # Backup listings, restore test results
├── vendor/                # SOC 2 reports from Stripe, Cloudflare, etc.
└── privacy/               # Privacy policy screenshots, consent flows
```

Create subdirectories as needed when collecting evidence.

## Collection Guide

See [../checklists/evidence-collection.md](../checklists/evidence-collection.md) for the full evidence inventory with source locations and collection instructions.
