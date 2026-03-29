# Enterprise Configuration Service

**Date:** 2026-03-29
**Status:** PLANNING
**Scope:** Every configurable behavior in the platform, organized into policy domains

## Design Principle

One configuration surface. Every "should this be on or off?" and "how strict should this be?" lives here. Admins don't hunt through 15 different settings pages. They open the configuration panel, select a domain, and adjust.

Team tier gets sensible defaults with minimal exposure. Enterprise tier gets full control.

## Architecture

```
Configuration Service
├── Policy Domains (categories of settings)
│   ├── Organization (branding, sites, roles)
│   ├── Quality (QMS rigor, scoping, thresholds)
│   ├── Process (graph, SPC, investigation behavior)
│   ├── Safety (HIRARC, observation frequency)
│   ├── Compliance (audit scheduling, document retention)
│   ├── Notifications (who gets notified of what)
│   └── Integration (API keys, webhooks, SSO)
│
├── Policy Presets (ISO 9001, IATF 16949, AS9100D, custom)
│   └── Apply preset → populates all domains with standard defaults
│
└── Per-Setting Structure
    ├── key: unique identifier (e.g., "quality.ncr.auto_create_investigation")
    ├── value: the setting value (bool, int, float, string, enum, json)
    ├── scope: tenant | site | user (which level this setting applies at)
    ├── tier_required: team | enterprise (minimum tier to configure)
    ├── default: what it is if never configured
    └── preset_overrides: what each preset sets it to
```

## Storage Model

```
TenantConfig
  tenant: FK → Tenant
  domain: CharField  # "quality", "process", "safety", etc.
  key: CharField     # "ncr.auto_create_investigation"
  value: JSONField   # any JSON-serializable value
  site: FK → Site (nullable)  # null = tenant-wide, set = site-specific override
  updated_by: FK → User
  updated_at: DateTimeField

  unique_together: (tenant, domain, key, site)
```

One table. One query pattern. Site-level overrides shadow tenant-level defaults:
1. Check `TenantConfig(tenant=X, domain=Y, key=Z, site=user_site)` — site override
2. If not found, check `TenantConfig(tenant=X, domain=Y, key=Z, site=null)` — tenant default
3. If not found, return hardcoded default from policy definition

## Policy Domains & Settings

---

### 1. ORGANIZATION

Settings that define the tenant's structure and identity.

| Key | Type | Default | Enterprise Only | Description |
|-----|------|---------|-----------------|-------------|
| `org.company_name` | string | "" | No | Company name for reports/exports |
| `org.logo_file_id` | uuid | null | No | Logo for PDF exports |
| `org.footer_text` | string | "" | No | Report footer text |
| `org.timezone` | string | "UTC" | No | Default timezone for date display |
| `org.date_format` | enum | "iso" | No | Date display format (iso/us/eu) |
| `org.fiscal_year_start` | int | 1 | Yes | Month number (1-12) fiscal year begins |
| `org.multi_site_enabled` | bool | false | Yes | Enable site dimension (auto-true for Enterprise) |
| `org.default_site` | uuid | null | Yes | Default site for new records when multi-site |

---

### 2. QUALITY — QMS Rigor & Behavior

Settings that control how strict the QMS operates. This is where ISO vs IATF vs lightweight diverge.

| Key | Type | Default | Enterprise Only | Description |
|-----|------|---------|-----------------|-------------|
| **NCR** | | | | |
| `quality.ncr.require_root_cause` | bool | true | No | Require root cause before CAPA transition |
| `quality.ncr.require_containment` | bool | false | No | Require containment action before investigation |
| `quality.ncr.auto_create_investigation` | bool | false | No | Auto-create Loop investigation when NCR opens |
| `quality.ncr.escalation_days` | int | 14 | No | Days before overdue NCR escalates to management |
| `quality.ncr.require_verification` | bool | true | No | Require verification step before closing |
| **CAPA** | | | | |
| `quality.capa.require_effectiveness_review` | bool | false | No | Require effectiveness review 90 days after close |
| `quality.capa.effectiveness_review_days` | int | 90 | No | Days after closure to check effectiveness |
| **FMEA** | | | | |
| `quality.fmea.methodology` | enum | "aiag_4th" | No | Scoring method: aiag_4th, svend_bayesian, svend_full |
| `quality.fmea.rpn_action_threshold` | int | 100 | No | RPN above which action is required |
| `quality.fmea.severity_threshold` | int | 8 | No | Severity above which action is required regardless of RPN |
| `quality.fmea.require_detection_control` | bool | true | No | Require detection control description on every row |
| **Supplier** | | | | |
| `quality.supplier.response_due_days` | int | 14 | No | Default days for supplier claim response |
| `quality.supplier.auto_escalate_rejections` | int | 2 | No | Number of rejected responses before auto-escalation suggestion |
| `quality.supplier.quality_score_threshold` | float | 0.5 | No | Response quality score below which reviewer is warned |
| **Audit** | | | | |
| `quality.audit.require_checklist` | bool | false | No | Require checklist attachment before completing audit |
| `quality.audit.finding_auto_ncr` | bool | false | No | Auto-create NCR from major/critical audit findings |
| `quality.audit.minimum_frequency_months` | int | 12 | No | Minimum audit frequency per ISO clause |
| **Training** | | | | |
| `quality.training.require_renewal` | bool | true | No | Require renewal tracking on training records |
| `quality.training.default_renewal_months` | int | 12 | No | Default renewal period |
| `quality.training.require_competency_assessment` | bool | false | Yes | Require assessment after training completion |
| **Document Control** | | | | |
| `quality.document.require_approval` | bool | true | No | Require approval before document becomes active |
| `quality.document.review_reminder_days` | int | 30 | No | Days before review due to send reminder |
| `quality.document.retention_years` | int | 7 | No | Default retention period |
| **Control Plan** | | | | |
| `quality.control_plan.require_reaction_plan` | bool | true | No | Require reaction plan on every control plan item |
| `quality.control_plan.require_gage_link` | bool | false | Yes | Require linked measurement equipment on items |
| **Scoping** | | | | |
| `quality.scoping.<model_name>` | enum | (per preset) | Yes | "tenant_wide", "site_scoped", "user_choice" — see configuration_service_spec.md |

---

### 3. PROCESS — Graph, SPC, Investigation

Settings that control how the process knowledge system behaves.

| Key | Type | Default | Enterprise Only | Description |
|-----|------|---------|-----------------|-------------|
| **Graph** | | | | |
| `process.graph.evidence_decay_half_life_days` | int | 180 | No | Recency weighting half-life for edge posteriors |
| `process.graph.contradiction_threshold` | float | 0.05 | No | P(evidence\|edge) below which contradiction signal fires (D8) |
| `process.graph.contradiction_cooldown_days` | int | 7 | No | Don't re-fire contradiction on same edge within this window |
| `process.graph.staleness_max_days` | int | 365 | No | Edges older than this are flagged stale |
| `process.graph.auto_seed_from_fmis` | bool | false | No | Auto-propose graph structure when FMIS rows created |
| **SPC** | | | | |
| `process.spc.auto_signal_on_ooc` | bool | false | No | Auto-create Signal when SPC detects out-of-control (off by default per D8) |
| `process.spc.signal_debounce_hours` | int | 8 | No | Minimum hours between auto-signals on same chart |
| `process.spc.flag_stale_edges` | bool | true | No | Flag graph edges as stale on SPC shift detection |
| `process.spc.default_rules` | list | ["1_beyond_3sigma"] | No | Default Western Electric / Nelson rules to apply |
| **Investigation** | | | | |
| `process.investigation.require_graph_scope` | bool | false | Yes | Require investigation to scope from graph (vs freeform) |
| `process.investigation.writeback_on_conclude` | bool | true | No | Prompt for graph writeback when investigation concludes |
| `process.investigation.max_duration_days` | int | 90 | No | Days before investigation flagged as overdue |
| **Process Confirmation** | | | | |
| `process.pc.require_photo` | bool | false | No | Require photo evidence on PC observations |
| `process.pc.minimum_frequency_per_week` | int | 0 | No | Minimum PCs per process area per week (0 = no minimum) |
| **Forced Failure Test** | | | | |
| `process.fft.minimum_injection_count` | int | 3 | No | Minimum failure injections per test |
| `process.fft.require_safety_review` | bool | true | No | Require safety review before conducting FFT |

---

### 4. SAFETY

Settings for HIRARC/safety program behavior.

| Key | Type | Default | Enterprise Only | Description |
|-----|------|---------|-----------------|-------------|
| `safety.observation_target_per_week` | int | 5 | No | Target Frontier Card observations per week per zone |
| `safety.require_immediate_action` | bool | true | No | Require immediate action on critical/high severity findings |
| `safety.auto_fmea_from_card` | bool | true | No | Auto-create FMEA row from Frontier Card findings |
| `safety.card_expiry_days` | int | 30 | No | Days before unresolved card escalates |

---

### 5. COMPLIANCE

Settings for audit and compliance automation.

| Key | Type | Default | Enterprise Only | Description |
|-----|------|---------|-----------------|-------------|
| `compliance.standard` | enum | "iso_9001" | No | Primary standard: iso_9001, iatf_16949, as9100d, custom |
| `compliance.audit_schedule_auto` | bool | false | Yes | Auto-generate audit schedule from clause coverage |
| `compliance.management_review_frequency_months` | int | 12 | No | Minimum management review frequency |
| `compliance.require_electronic_signatures` | bool | false | Yes | Require e-signatures on controlled documents |
| `compliance.change_request_required` | bool | true | No | Require change request for document revisions |

---

### 6. NOTIFICATIONS

Settings for notification behavior. These determine who gets notified of what.

| Key | Type | Default | Enterprise Only | Description |
|-----|------|---------|-----------------|-------------|
| `notifications.ncr_created` | list | ["owner", "site_admin"] | No | Who to notify on NCR creation |
| `notifications.investigation_overdue` | list | ["owner", "sponsor"] | No | Who to notify on overdue investigation |
| `notifications.commitment_due_soon_days` | int | 3 | No | Days before due to notify commitment owner |
| `notifications.signal_created` | list | ["site_admin"] | No | Who to notify on new signal |
| `notifications.graph_contradiction` | list | ["process_owner"] | No | Who to notify on graph contradiction |
| `notifications.supplier_response_received` | list | ["claim_owner"] | No | Who to notify when supplier responds |
| `notifications.calibration_due_days` | int | 30 | No | Days before calibration due to notify |
| `notifications.training_expiry_days` | int | 30 | No | Days before training expires to notify |

---

### 7. INTEGRATION

Settings for external system integration. Mostly Enterprise.

| Key | Type | Default | Enterprise Only | Description |
|-----|------|---------|-----------------|-------------|
| `integration.api_keys_enabled` | bool | false | Yes | Allow API key authentication |
| `integration.webhook_enabled` | bool | false | Yes | Allow outbound webhooks |
| `integration.sso_provider` | enum | "none" | Yes | SSO provider: none, saml, oidc |
| `integration.sso_config` | json | {} | Yes | SSO configuration blob |
| `integration.export_format` | enum | "pdf" | No | Default export format: pdf, xlsx, docx |

---

## Preset Definitions

When an admin selects a preset, it populates settings across ALL domains.

### ISO 9001 Preset

```python
ISO_9001 = {
    "compliance.standard": "iso_9001",
    "quality.ncr.require_root_cause": True,
    "quality.ncr.require_verification": True,
    "quality.fmea.methodology": "aiag_4th",
    "quality.fmea.rpn_action_threshold": 100,
    "quality.audit.minimum_frequency_months": 12,
    "quality.document.require_approval": True,
    "quality.document.retention_years": 7,
    "quality.training.require_renewal": True,
    "compliance.management_review_frequency_months": 12,
    "quality.control_plan.require_reaction_plan": True,
    # Scoping defaults
    "quality.scoping.ncr": "site_scoped",
    "quality.scoping.fmea": "site_scoped",
    "quality.scoping.audit": "site_scoped",
    "quality.scoping.equipment": "site_scoped",
    "quality.scoping.control_plan": "site_scoped",
    "quality.scoping.complaint": "user_choice",
    "quality.scoping.signal": "site_scoped",
    "quality.scoping.commitment": "site_scoped",
    "quality.scoping.investigation": "site_scoped",
}
```

### IATF 16949 Preset

```python
IATF_16949 = {
    **ISO_9001,
    "compliance.standard": "iatf_16949",
    "quality.fmea.require_detection_control": True,
    "quality.control_plan.require_gage_link": True,
    "quality.capa.require_effectiveness_review": True,
    "quality.capa.effectiveness_review_days": 90,
    "quality.ncr.require_containment": True,
    "quality.supplier.auto_escalate_rejections": 2,
    "quality.audit.require_checklist": True,
    "quality.audit.finding_auto_ncr": True,
    "process.fft.require_safety_review": True,
    "compliance.require_electronic_signatures": True,
}
```

### AS9100D Preset

```python
AS9100D = {
    **ISO_9001,
    "compliance.standard": "as9100d",
    "quality.ncr.require_containment": True,
    "quality.ncr.escalation_days": 7,  # aerospace = faster
    "quality.document.retention_years": 10,  # longer retention
    "compliance.require_electronic_signatures": True,
    "quality.training.require_competency_assessment": True,
    "process.fft.minimum_injection_count": 5,  # more rigorous
}
```

### Lightweight / Startup Preset

```python
LIGHTWEIGHT = {
    "compliance.standard": "custom",
    "quality.ncr.require_root_cause": False,
    "quality.ncr.require_verification": False,
    "quality.fmea.methodology": "aiag_4th",
    "quality.fmea.rpn_action_threshold": 200,  # higher = less action
    "quality.audit.minimum_frequency_months": 0,  # no minimum
    "quality.document.require_approval": False,
    "quality.training.require_renewal": False,
    "process.graph.staleness_max_days": 0,  # no staleness tracking
    "process.spc.auto_signal_on_ooc": False,
}
```

## UI — Configuration Panel

The configuration panel replaces scattered settings pages with one surface:

```
┌──────────────────────────────────────────────────────────────┐
│  Configuration                              [ISO 9001 ▾]     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────────────────────────────┐  │
│  │ Organization │  │  NCR Settings                        │  │
│  │ ● Quality    │  │                                      │  │
│  │   Process    │  │  Require root cause         [✓]      │  │
│  │   Safety     │  │  Require containment        [ ]      │  │
│  │   Compliance │  │  Auto-create investigation  [ ]      │  │
│  │   Notify     │  │  Escalation days           [14]      │  │
│  │   Integration│  │  Require verification       [✓]      │  │
│  │              │  │                                      │  │
│  │ Quality:     │  │  FMEA Settings                      │  │
│  │  ● NCR       │  │                                      │  │
│  │    CAPA      │  │  Methodology       [AIAG 4th Ed ▾]  │  │
│  │    FMEA      │  │  RPN action threshold      [100]     │  │
│  │    Supplier   │  │  Severity threshold         [8]     │  │
│  │    Audit     │  │  Require detection ctrl     [✓]      │  │
│  │    Training  │  │                                      │  │
│  │    Document  │  │  Supplier Settings                   │  │
│  │    Control   │  │                                      │  │
│  │    Plan      │  │  Response due days         [14]      │  │
│  │    Scoping   │  │  Auto-escalate after       [2] rej   │  │
│  │              │  │  Quality score warning     [0.5]     │  │
│  └──────────────┘  └──────────────────────────────────────┘  │
│                                                              │
│  ⓘ ISO 9001 preset applied. Changes marked with •           │
│  [Reset to Preset]                     [Save Changes]        │
└──────────────────────────────────────────────────────────────┘
```

Left sidebar: domain categories with subsections.
Right panel: settings for selected subsection.
Preset dropdown at top: apply a standard template.
Changed settings marked with a dot so admin knows what they've customized.

## Service Interface

```python
class ConfigService:
    """Read/write tenant configuration."""

    @staticmethod
    def get(tenant_id, key, site_id=None) -> any:
        """Get a config value. Site override → tenant default → hardcoded default."""

    @staticmethod
    def set(tenant_id, key, value, site_id=None, updated_by=None):
        """Set a config value. Creates or updates TenantConfig row."""

    @staticmethod
    def get_domain(tenant_id, domain, site_id=None) -> dict:
        """Get all settings in a domain as a dict."""

    @staticmethod
    def apply_preset(tenant_id, preset_name, updated_by=None):
        """Apply a preset — creates/updates TenantConfig rows for all preset keys."""

    @staticmethod
    def get_changed_from_preset(tenant_id) -> list:
        """List settings that differ from the active preset."""

    @staticmethod
    def reset_to_preset(tenant_id, preset_name, updated_by=None):
        """Reset all settings to preset defaults (destructive)."""
```

## How Other Services Consume This

Every service that has configurable behavior reads from ConfigService:

```python
# In loop/views.py — NCR creation
if ConfigService.get(tenant_id, "quality.ncr.auto_create_investigation"):
    Investigation.objects.create(...)

# In graph/service.py — evidence stacking
half_life = ConfigService.get(tenant_id, "process.graph.evidence_decay_half_life_days")
posterior = _recompute_posterior(edge, half_life_days=half_life)

# In loop/response_quality.py — supplier thresholds
threshold = ConfigService.get(tenant_id, "quality.supplier.quality_score_threshold")

# In graph/integrations.py — SPC auto-signal
if ConfigService.get(tenant_id, "process.spc.auto_signal_on_ooc"):
    Signal.objects.create(...)
```

## Implementation Order

1. **TenantConfig model** — one table, simple schema
2. **ConfigService** — get/set/get_domain/apply_preset
3. **Preset constants** — ISO 9001, IATF, AS9100D, Lightweight
4. **Wire critical paths** — graph decay, contradiction threshold, SPC auto-signal (the settings that already exist as hardcoded values)
5. **Configuration panel UI** — the admin surface
6. **Onboarding flow** — "Select your standard" step that applies a preset
7. **Remaining settings** — wire everything else incrementally

Steps 1-3 are foundation. Step 4 replaces hardcoded values with ConfigService.get() calls. Steps 5-7 are UX polish.
