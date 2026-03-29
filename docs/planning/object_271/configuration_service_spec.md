# Configuration Service Spec — Tenant & Site Scoping

**Date:** 2026-03-29
**Owner:** S1 (spec), S2 (model plumbing)
**Depends on:** S2's tenant/site FK mapping (in progress)

## Tier Constraint

| Tier | Sites | Scoping behavior |
|------|-------|-----------------|
| **Team ($99/mo)** | 1 | Everything is tenant-wide. Site dimension doesn't exist. No site picker, no site filter, no scoping config. Site FK always null. |
| **Enterprise ($299/mo)** | N | Site scoping activates. ScopingRule determines per-model behavior. Site picker in creation forms. Policy panel shows scoping config. |

**The configuration service is Enterprise-only.** Team customers never see it. Their records are all tenant-wide by definition (single site = no scoping needed).

Site FKs are added to all models regardless of tier (plumbing). For Team tier they're always null. The FK exists for upgrade path.

## Problem (Enterprise Only)

Multi-site enterprises need two dimensions of scoping:
- **Tenant** — security boundary (who CAN see it)
- **Site** — organizational scope (who SHOULD see it)

But real orgs need flexibility:
- A major NCR should be visible across sites even though NCRs are normally site-scoped
- Some records MUST be site-scoped (ISO clause 4.3 scope), others MUST be org-wide (management review)
- IATF 16949 has different scoping requirements than ISO 9001

The configuration service lets Enterprise admins define these rules.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Configuration Service                │
│                                                     │
│  ScopingRule: per-model scoping policy              │
│  VisibilityOverride: per-record exception           │
│  PolicyPreset: ISO 9001 / IATF 16949 / custom      │
│                                                     │
│  Consumed by:                                       │
│  - TenantIsolationMiddleware (security)             │
│  - View layer (site filtering)                      │
│  - GraphService (graph scoping)                     │
│  - Loop policy panel (UI)                           │
└─────────────────────────────────────────────────────┘
```

## Models

### ScopingRule

Defines the default scoping behavior for a model within a tenant.

```python
class ScopingRule(models.Model):
    """Per-model scoping policy for a tenant.

    Determines whether records of a given type are site-scoped,
    tenant-wide, or user-configurable at creation time.
    """
    class ScopeType(models.TextChoices):
        TENANT_WIDE = "tenant_wide", "Tenant-Wide (all sites see it)"
        SITE_SCOPED = "site_scoped", "Site-Scoped (only assigned site)"
        USER_CHOICE = "user_choice", "User Chooses at Creation"

    tenant = FK → Tenant
    model_name = CharField(max_length=100)  # e.g., "ncr", "fmea", "investigation"
    scope_type = CharField(choices=ScopeType)
    default_site = FK → Site (nullable)  # auto-assign when user_choice not exercised
    created_at = DateTimeField
    updated_at = DateTimeField

    class Meta:
        unique_together = [("tenant", "model_name")]
```

### VisibilityOverride

Per-record exception to the scoping rule. "This specific NCR should be visible to all sites even though NCRs are normally site-scoped."

```python
class VisibilityOverride(models.Model):
    """Per-record visibility exception.

    Allows promoting a site-scoped record to tenant-wide visibility,
    or restricting a tenant-wide record to specific sites.
    """
    class OverrideType(models.TextChoices):
        PROMOTE_TO_TENANT = "promote", "Promote to Tenant-Wide"
        RESTRICT_TO_SITES = "restrict", "Restrict to Specific Sites"

    tenant = FK → Tenant
    content_type = FK → ContentType  # generic FK to any model
    object_id = UUIDField
    override_type = CharField(choices=OverrideType)
    sites = M2M → Site (blank=True)  # for restrict: which sites can see it
    reason = TextField(blank=True)  # why the override exists
    created_by = FK → User
    created_at = DateTimeField
```

### PolicyPreset

Pre-configured scoping rules for common standards. An admin selects "ISO 9001" and gets sensible defaults. Can customize from there.

```python
# Not a Django model — a Python dict in a constants file
PRESETS = {
    "iso_9001": {
        "ncr": "site_scoped",
        "capa": "site_scoped",
        "fmea": "site_scoped",
        "investigation": "site_scoped",
        "internal_audit": "site_scoped",
        "training_requirement": "tenant_wide",  # org-wide competency matrix
        "management_review": "tenant_wide",
        "controlled_document": "tenant_wide",  # one set of controlled docs
        "supplier": "tenant_wide",  # suppliers serve the whole org
        "hoshin_project": "tenant_wide",
        "equipment": "site_scoped",  # calibration is per-plant
        "control_plan": "site_scoped",
        "process_confirmation": "site_scoped",
        "signal": "site_scoped",
        "commitment": "site_scoped",
        "supplier_claim": "tenant_wide",  # claims are org-level
        "process_graph": "tenant_wide",  # one graph per org (GRAPH-001)
    },
    "iatf_16949": {
        # Inherits ISO 9001 defaults, overrides where IATF differs
        **"iso_9001",
        "control_plan": "site_scoped",  # IATF requires site-specific control plans
        "fmea": "site_scoped",
    },
    "single_site": {
        # Everything tenant-wide — site dimension is unused
        "_default": "tenant_wide",
    },
}
```

## Service Interface

```python
class ScopingService:
    """Configuration service for tenant/site scoping."""

    @staticmethod
    def get_scope(tenant_id, model_name) -> str:
        """Return scoping type for a model within a tenant.
        Returns: "tenant_wide", "site_scoped", or "user_choice".
        Falls back to preset or "tenant_wide" if no rule exists.
        """

    @staticmethod
    def apply_preset(tenant_id, preset_name):
        """Apply a preset (e.g., "iso_9001") to create ScopingRules."""

    @staticmethod
    def filter_queryset(queryset, user, model_name):
        """Apply scoping to a queryset based on rules + user's site access.

        1. Tenant isolation (always)
        2. If site_scoped: filter to user's assigned sites
        3. If tenant_wide: no site filter
        4. Apply visibility overrides (promoted/restricted records)
        """

    @staticmethod
    def check_visibility(user, obj) -> bool:
        """Can this user see this specific record?
        Checks: tenant membership, site assignment, visibility overrides.
        """

    @staticmethod
    def get_overrides(tenant_id, model_name) -> list:
        """List all visibility overrides for a model in a tenant."""
```

## UI — Policy Panel Integration

The scoping configuration lives in the Loop policy panel (LOOP-001 §4). Admins see:

```
┌─────────────────────────────────────────────────────────┐
│  Record Scoping                                   [ISO 9001 ▾]  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Record Type          Scope              Default Site   │
│  ─────────────        ──────────────     ────────────   │
│  NCR                  ⊙ Site-Scoped      [Auto]         │
│  CAPA                 ⊙ Site-Scoped      [Auto]         │
│  FMEA                 ⊙ Site-Scoped      [Auto]         │
│  Investigation        ⊙ Site-Scoped      [Auto]         │
│  Internal Audit       ⊙ Site-Scoped      [Auto]         │
│  Training             ○ Tenant-Wide      —              │
│  Documents            ○ Tenant-Wide      —              │
│  Management Review    ○ Tenant-Wide      —              │
│  Suppliers            ○ Tenant-Wide      —              │
│  Equipment            ⊙ Site-Scoped      [Auto]         │
│  Control Plan         ⊙ Site-Scoped      [Auto]         │
│  Process Graph        ○ Tenant-Wide      —              │
│  Hoshin               ○ Tenant-Wide      —              │
│  Supplier Claims      ○ Tenant-Wide      —              │
│  Signals              ⊙ Site-Scoped      [Auto]         │
│  Commitments          ⊙ Site-Scoped      [Auto]         │
│                                                         │
│  [Reset to ISO 9001 Defaults]    [Save Changes]        │
└─────────────────────────────────────────────────────────┘
```

The preset dropdown at top-right applies a standard template. Admins can then customize individual rows. Changes are saved as ScopingRules.

## How It's Used

### At Record Creation

When a user creates an NCR:
1. `ScopingService.get_scope(tenant_id, "ncr")` → "site_scoped"
2. If "site_scoped" and user has one site → auto-assign
3. If "site_scoped" and user has multiple sites → show site picker
4. If "tenant_wide" → no site assignment
5. If "user_choice" → show radio: "This site only" / "All sites"

### At Query Time

When listing NCRs:
1. `ScopingService.filter_queryset(NCR.objects.all(), user, "ncr")`
2. Applies tenant filter (always)
3. Checks scoping rule → "site_scoped"
4. Filters to user's assigned sites
5. Adds any promoted records (VisibilityOverride)
6. Removes any restricted records

### At Override Time

Quality manager says "this NCR affects all plants":
1. Create VisibilityOverride(type="promote", object_id=ncr.id)
2. NCR now appears in all site views
3. Override reason logged for audit trail

## Model Scoping Map

Every business model mapped to its natural scoping default. "Configurable" means Enterprise admins can change it via the policy panel.

### Tier 1: Always Tenant-Wide (not configurable)

These are structurally org-wide. Site scoping makes no sense.

| Model | App | Why tenant-wide |
|-------|-----|----------------|
| Tenant | core | IS the tenant |
| Membership | core | Org membership |
| Employee | agents_api | Personnel records span sites |
| SupplierRecord | agents_api | Suppliers serve the whole org |
| SupplierClaim | loop | Claims are org-level procurement |
| HoshinProject | agents_api | Strategic planning is org-wide |
| ResourceCommitment | agents_api | Resource allocation is org-wide |
| ManagementReview | agents_api | ISO 9001 §9.3 — org-level |
| ManagementReviewTemplate | agents_api | Org-level templates |
| ControlledDocument | agents_api | One set of controlled docs per org |
| ProcessGraph | graph | GRAPH-001: one graph per org (federated via process_area) |
| QMSPolicy | loop | Policies apply org-wide |
| TrainingRequirement | agents_api | Competency matrix is org-wide |
| AuditorPortalToken | loop | Auditor access is org-level |
| ISODocument | agents_api | Authored documents are org-level |

### Tier 2: Default Site-Scoped (configurable for Enterprise)

These naturally belong to a specific plant/location. Enterprise admins can promote to tenant-wide.

| Model | App | Why site-scoped | Configurable? |
|-------|-----|----------------|---------------|
| NonconformanceRecord | agents_api | NCRs are plant-specific defects | Yes — can promote |
| CustomerComplaint | agents_api | Complaints may be product/site-specific | Yes — can promote |
| InternalAudit | agents_api | Audits are per-site (ISO 9001 §9.2) | Yes — can promote |
| FMEA / FMEARow | agents_api | Process FMEA is per-process, per-site | Yes — can promote |
| FMIS / FMISRow | loop | Same as FMEA | Yes |
| MeasurementEquipment | agents_api | Equipment lives at a site | Yes — can promote |
| ControlPlan / ControlPlanItem | agents_api | Control plans are per-process, per-site | Yes — can promote |
| ProcessConfirmation | loop | Gemba observations are on-site | No — always site |
| ForcedFailureTest | loop | Detection testing is on-site | No — always site |
| Signal | loop | Signals originate at a site | Yes — can promote |
| Commitment | loop | Commitments are usually site-level work | Yes — can promote |
| Investigation | core | Investigations scope to a process/site | Yes — can promote |
| FrontierZone | safety | Safety zones are physical locations | No — always site |
| FrontierCard | safety | Safety observations are on-site | No — always site |
| AuditSchedule | safety | Safety audit schedules are per-site | No — always site |

### Tier 3: Default User-Scoped (not site-relevant)

These belong to individual users, not sites or tenants.

| Model | App | Why user-scoped |
|-------|-----|----------------|
| Project | core | Personal analysis projects |
| Hypothesis | core | Within a project |
| Evidence | core | Within a project |
| Dataset | core | Uploaded by user |
| ExperimentDesign | core | Within a project |
| DSWResult | agents_api | Analysis result |
| SavedModel | agents_api | ML model |
| Board / Whiteboard | agents_api | Personal collaboration |
| Workbench | workbench | Personal workspace |
| UserFile | files | Personal uploads |
| Notebook | ? | Personal field book |

### Tier 4: Derived Scope (inherits from parent)

These don't need their own scoping — they inherit from their parent record.

| Model | App | Inherits from |
|-------|-----|--------------|
| CAPAReport | agents_api | NCR |
| AuditFinding | agents_api | InternalAudit |
| AuditChecklist | agents_api | InternalAudit |
| TrainingRecord | agents_api | TrainingRequirement |
| DocumentRevision | agents_api | ControlledDocument |
| Risk | agents_api | Site or tenant (configurable) |
| ModeTransition | loop | Investigation/Commitment |
| CommitmentNote | loop | Commitment |
| CommitmentResource | loop | Commitment |
| PCObservationItem | loop | ProcessConfirmation |
| InvestigationEntry | loop | Investigation |
| SupplierResponse | loop | SupplierClaim |
| ClaimVerification | loop | SupplierClaim |
| PolicyCondition | loop | QMSPolicy |
| TrainingReflection | loop | TrainingRequirement context |
| ProcessNode | graph | ProcessGraph |
| ProcessEdge | graph | ProcessGraph |
| EdgeEvidence | graph | ProcessEdge |
| AFE | agents_api | HoshinProject |
| AFEApprovalLevel | agents_api | AFE |

## How filter_queryset() Works

```python
def filter_queryset(qs, user, model_name):
    tenant = user.active_tenant

    # 1. Tenant isolation (always, all tiers)
    qs = qs.filter(tenant=tenant)  # or via site__tenant for site-only models

    # 2. Team tier: done. No site filtering.
    if not tenant.is_enterprise:
        return qs

    # 3. Enterprise: check scoping rule
    scope = ScopingService.get_scope(tenant.id, model_name)

    if scope == "tenant_wide":
        return qs  # no site filter

    if scope == "site_scoped":
        user_sites = user.site_access.values_list("site_id", flat=True)
        qs = qs.filter(
            Q(site_id__in=user_sites) |
            Q(id__in=promoted_record_ids)  # visibility overrides
        )
        return qs

    if scope == "user_choice":
        # Records that were created as "all sites" have site=null
        # Records created as site-specific have site set
        user_sites = user.site_access.values_list("site_id", flat=True)
        qs = qs.filter(
            Q(site__isnull=True) |  # tenant-wide by creator's choice
            Q(site_id__in=user_sites)  # site-scoped
        )
        return qs
```

## What S2 Lays Down (Prerequisites)

S2's model mapping adds:
- `tenant` FK to every model that needs it (security boundary)
- `site` FK (nullable) to every model where site scoping is possible
- These FKs exist whether or not the scoping rule activates them

The configuration service sits on TOP of those FKs. The FKs are the pipe. The service is the valve.

## What We Don't Build Yet

- Multi-site visibility rules (record visible to sites A and C but not B)
- Role-based scoping (managers see all sites, operators see only theirs) — this is handled by existing SiteAccess model
- Automatic site assignment from user's primary site — simple, do when needed
- Scoping rule versioning/history — not needed until enterprise customers request it

## Implementation Order

1. **S2 (now):** Add tenant/site FKs to all models. Migration. This is plumbing.
2. **Later:** ScopingRule + ScopingService models and service class.
3. **Later:** Policy panel UI for scoping configuration.
4. **Later:** Wire ScopingService.filter_queryset() into view layer.
5. **Later:** VisibilityOverride model and UI.
6. **Later:** Preset templates (ISO 9001, IATF 16949, single-site).

Step 1 is the only one that touches the database now. Steps 2-6 are additive and can be built incrementally.

## How Presets Work

When an Enterprise admin selects a preset (e.g., "ISO 9001"), the system:
1. Creates ScopingRules for every Tier 2 model with the preset's defaults
2. Tier 1 models are not configurable — always tenant-wide
3. Tier 3 models are not configurable — always user-scoped
4. Tier 4 models inherit — no rules needed

The admin can then customize individual rules. The preset is the starting point, not a constraint.

### ISO 9001 Preset

| Model | Default | Rationale |
|-------|---------|-----------|
| NCR | site_scoped | §10.2 — nonconformity is local |
| Complaint | user_choice | §9.1.2 — product complaints may span sites |
| Audit | site_scoped | §9.2 — internal audits are per-site |
| FMEA/FMIS | site_scoped | §6.1 — risk assessment per process |
| Equipment | site_scoped | §7.1.5 — calibration per plant |
| Control Plan | site_scoped | §8.5.1 — process control per line |
| Signal | site_scoped | Signals originate at a site |
| Commitment | site_scoped | Work happens at a site |
| Investigation | site_scoped | Investigations scope to a process |

### IATF 16949 Preset

Inherits ISO 9001, adds:
- Control Plan: **always site_scoped** (IATF §8.5.1.1 — cannot be tenant-wide)
- FMEA: **always site_scoped** (IATF — process FMEA is per-manufacturing-process)
- Equipment: **always site_scoped** (MSA requirements are per-plant)

### Single-Site / Team Preset

All Tier 2 models → tenant_wide. This is the automatic default for Team tier. No UI exposed.

## Graph Interaction

ProcessGraph is always tenant-wide (Tier 1). But site scoping interacts with the graph at the NODE level:

- When an FMEA is site-scoped, its seeded ProcessNodes get tagged with `process_area` matching the site name
- This is GRAPH-001 D4's federated schema: `ProcessGraph.process_area` and `ProcessNode.shared`
- A node tagged `process_area="plant_a"` is visible in Plant A's graph view lens
- A shared node (`shared=True`) appears in all site views

So: the graph is ONE structure (tenant-wide), but VIEWS into it are site-filtered via process_area tags. The ScopingService doesn't filter graph queries directly — the graph's own lens system (GRAPH-001 §15.2) handles it.

## Open Questions

1. Should ScopingRule live in `loop/` (policy domain) or a new `config/` app?
   - **Leaning loop/** — it's a QMS Policy concern, and the policy panel already lives in loop.
2. Should presets be Django models (admin-editable) or Python constants (developer-maintained)?
   - **Leaning constants** — presets are standards-defined, not org-customizable. The RULES are customizable, not the presets themselves.
3. Do we need "user_choice" scope type, or is site_scoped + VisibilityOverride sufficient?
   - **Leaning yes** — some orgs want the creator to decide per-record, not per-model.
4. When a Team customer upgrades to Enterprise, do they get a one-time "assign your records to sites" wizard?
   - **Probably yes** — all existing records have site=null, they need to bulk-assign.
