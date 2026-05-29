# SVEND Container System Design

> **3P Stage:** Requirements → Design (this document) → Build
> **Date:** 2026-05-04
> **Requirements:** `docs/planning/3P_REQUIREMENTS_SPEC.md`
> **Build approach:** Workflow-first. New apps alongside existing. Gradual phase-out.
> **Plugin:** `svend@eric-tools` — built with plugin-dev, dog-foods the facilitation layer.

---

## 1. Architecture Overview

Two new Django apps (`workflows/`, `governance/`) + extend existing `pcl/` + Claude Code plugin.

**Build order:** workflows → governance → PCL (diff/merge existing) → Claude plugin (svend@eric-tools)

**Principle:** Everything inherits SynaraEntity. No exceptions. Clean as we go — any model we touch that doesn't inherit SynaraEntity gets fixed in the same CR.

**Coexistence:** Old apps (agents_api, etc.) stay live. New system runs parallel. Phase out when container + ecosystem is developed enough and reliable.

**Control phase:** Testing, calibration, standardization, and compliance automation close the loop after build.

---

## 2. App Structure

```
~/kjerne/
├── workflows/                  # NEW — workflow engine
│   ├── __init__.py
│   ├── models.py               # WorkflowDef, WorkflowInstance, WorkflowStep, Heuristic
│   ├── engine.py               # Execution, retention, heuristic matching
│   ├── templates.py            # Pre-built workflow templates
│   ├── serializers.py          # JSON serialization for API + export
│   ├── views.py                # API endpoints
│   ├── urls.py
│   ├── admin.py
│   └── tests/
│       ├── __init__.py
│       ├── test_models.py
│       ├── test_engine.py
│       ├── test_templates.py
│       └── test_views.py
│
├── pcl/                        # DEPLOYED — Process Characteristics Library (hybrid update)
│   ├── __init__.py
│   ├── models.py               # Measure, Datapoint
│   ├── resolution.py           # Calculated measure resolution, staleness detection
│   ├── confidence.py           # Auto-computed confidence from source_type x n
│   ├── views.py                # API endpoints
│   ├── urls.py
│   ├── admin.py
│   └── tests/
│       ├── __init__.py
│       ├── test_models.py
│       ├── test_resolution.py
│       ├── test_confidence.py
│       └── test_views.py
│
├── governance/                 # NEW — Synara governance engine
│   ├── __init__.py
│   ├── models.py               # GovernanceRule, GovernanceDecision, GovernanceOutcome, Contract
│   ├── confidence.py           # Bayesian confidence update from outcomes
│   ├── views.py                # API endpoints
│   ├── urls.py
│   ├── admin.py
│   └── tests/
│
├── syn/                        # PRESERVED — infrastructure layer (no model changes)
│   ├── synara/                 # PRESERVED as middleware only (NOT a Django app)
│   │   └── middleware/         # tenant, CSP, rate limit
│   └── (audit, core, err, log, api, sched, varta — ALL PRESERVED)
│
└── (accounts, core, agents_api, analysis, qms, prova — ALL PRESERVED)
```

---

## 3. Data Models

All models inherit `SynaraEntity` which provides: UUID PK, `correlation_id`, `tenant_id`, `created_at`/`updated_at`, `created_by`/`updated_by`, `is_deleted`/`deleted_at`/`deleted_by`, `metadata` (JSON), event emission hooks, soft delete manager.

Only domain-specific fields shown below.

### 3.1 workflows/models.py

```python
from syn.core.base_models import SynaraEntity


class WorkflowDef(SynaraEntity):
    """
    Reusable pipeline definition. The 'how we do this' template.
    
    Source 'learned' = auto-captured from user actions.
    Source 'system' = ships with the product (Tier 1).
    Source 'marketplace' = add-in (Tier 2).
    Source 'user' = manually created by user.
    
    Marketplace items are user workflows promoted to public visibility.
    
    REQ-WF-1, REQ-WF-4, REQ-WF-5, REQ-WF-6, REQ-WF-7
    """
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    schema = models.JSONField(default=dict)
    source = models.CharField(
        max_length=20,
        choices=[
            ('system', 'System'),
            ('user', 'User'),
            ('marketplace', 'Marketplace'),
            ('learned', 'Learned'),
        ],
    )
    visibility = models.CharField(
        max_length=20,
        choices=[
            ('private', 'Private'),
            ('org', 'Organization'),
            ('public', 'Public'),
        ],
        default='private',
    )
    version = models.IntegerField(default=1)
    parent = models.ForeignKey(
        'self', null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='forks',
    )
    tags = models.JSONField(default=list)
    tier = models.CharField(
        max_length=10,
        choices=[('free', 'Free'), ('addon', 'Add-on'), ('claude', 'Claude')],
        default='free',
    )

    class Meta:
        ordering = ['-updated_at']


class WorkflowInstance(SynaraEntity):
    """
    A running/completed execution of a workflow. The actual work.
    
    workflow_def is null for ad-hoc work (not from template).
    Ad-hoc instances can be promoted to WorkflowDef after completion
    (source='learned').
    
    REQ-IN-1, REQ-IN-2, REQ-IN-3, REQ-IN-4
    """
    workflow_def = models.ForeignKey(
        WorkflowDef, null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='instances',
    )
    name = models.CharField(max_length=200)
    status = models.CharField(
        max_length=20,
        choices=[
            ('active', 'Active'),
            ('paused', 'Paused'),
            ('completed', 'Completed'),
            ('failed', 'Failed'),
        ],
        default='active',
    )
    context = models.JSONField(default=dict)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    is_locked = models.BooleanField(default=False)  # REQ-IN-4 validated state flag
    locked_at = models.DateTimeField(null=True, blank=True)
    locked_by = models.CharField(max_length=200, blank=True)
    depth = models.PositiveSmallIntegerField(default=0)  # Max depth guard
    parent_step = models.ForeignKey(
        'WorkflowStep', null=True, blank=True,
        on_delete=models.PROTECT,
        related_name='child_instances',
    )
    MAX_DEPTH = 3  # Prevent recursive execution

    class Meta:
        ordering = ['-started_at']


class WorkflowStep(SynaraEntity):
    """
    One step in an instance execution. The provenance trail.
    
    This IS the analytical provenance — every step records what went in,
    what came out, which tool was used, and how long it took.
    The chain of steps IS the reasoning trail auditors want.
    
    REQ-IN-2, HC-18 (analytical provenance)
    """
    instance = models.ForeignKey(
        WorkflowInstance,
        on_delete=models.PROTECT,  # Red/blue #4: provenance must survive instance deletion
        related_name='steps',
    )
    sequence = models.IntegerField()
    action = models.CharField(max_length=100)
    inputs = models.JSONField(default=dict)
    outputs = models.JSONField(default=dict)
    tool = models.CharField(max_length=100, blank=True)
    tool_version = models.CharField(max_length=20, blank=True)
    duration_ms = models.IntegerField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('running', 'Running'),
            ('completed', 'Completed'),
            ('failed', 'Failed'),
            ('skipped', 'Skipped'),
        ],
        default='pending',
    )
    idempotency_key = models.CharField(max_length=64, unique=True)  # Crash recovery
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error = models.TextField(blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['instance', 'sequence']
        unique_together = [('instance', 'sequence')]


class Heuristic(SynaraEntity):
    """
    A learned pattern recognized from session logs — NOT user-created.
    
    Heuristics are born when the system detects repeated session patterns.
    Synara logs every session immutably. The heuristic engine matches on:
    - data_schema: column types, ranges, structure of uploaded/selected data
    - temporal_pattern: user + day/time + frequency (e.g., Lisa + Thursday)
    - role_pattern: users with this role + this schema → usually this workflow
    - session_source: UUID chain to the originating session logs
    
    Confidence updates via Bayesian reinforcement:
    - Accepted + confirmed outcome → confidence up
    - Dismissed → confidence down  
    - Decay over time if unused
    
    Surfacing: fires only when confidence > threshold.
    - Paid tier: Claude reads heuristic, proposes workspace setup
    - Free tier: quick-action bar shows suggestion
    
    Promotion: implicit heuristic can be promoted to explicit WorkflowDef
    by adding triggers, conditionals, governance rules, schedule.
    
    REQ-WF-2, REQ-WF-3, REQ-SY-1, REQ-SY-2
    """
    # Optional: promoted heuristics link to a formal WorkflowDef
    workflow_def = models.ForeignKey(
        WorkflowDef, null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='heuristics',
    )
    name = models.CharField(max_length=200)
    
    # Matching criteria (what triggers this heuristic)
    data_schema = models.JSONField(default=dict)      # column types, ranges, structure
    temporal_pattern = models.JSONField(default=dict)  # day_of_week, frequency, time_window
    role_pattern = models.CharField(max_length=100, blank=True)  # operator, qe, lean_ci, etc.
    trigger = models.JSONField(default=dict)           # additional match conditions
    
    # What to propose
    suggestion = models.JSONField(default=dict)        # workspace config: tools, bindings, layout
    
    # Source provenance — which session logs generated this heuristic
    session_source = models.JSONField(default=list)    # list of SynaraImmutableLog UUIDs
    
    # Bayesian confidence tracking
    confidence = models.FloatField(default=0.5)
    confidence_threshold = models.FloatField(default=0.7)  # only fire above this
    times_fired = models.IntegerField(default=0)
    times_accepted = models.IntegerField(default=0)
    times_dismissed = models.IntegerField(default=0)
    last_fired = models.DateTimeField(null=True, blank=True)
    
    # Staleness — for explicit/promoted workflows
    staleness_check = models.JSONField(null=True, blank=True)  # PCL fields to check for updates
    schedule = models.CharField(max_length=100, blank=True)    # cron expression for scheduled runs
    
    # Governance — for explicit/promoted workflows  
    requires_approval = models.BooleanField(default=False)
    approval_condition = models.JSONField(null=True, blank=True)  # e.g., {"if": "cpk < 1.33"}

    class Meta:
        ordering = ['-confidence', '-times_accepted']
```

### 3.2 pcl/ — DIFF/MERGE WITH EXISTING

**⚠ RED/BLUE FINDING (2026-05-04): PCL already exists at ~/kjerne/pcl/ with applied
migration 0001_initial. The deployed schema differs from this spec. Resolution:
diff and merge — take the best of both schemas.**

Existing pcl/ has:
- `Measure`: uses `slug` (not `name`), `range_min`/`range_max`, formula references via `[slug]` syntax
- `Datapoint`: extends `SynaraImmutableLog` (hash-chained, write-once), NOT `SynaraEntity`
- Confidence: log2-based curves with per-source-type plateau values (more sophisticated)
- Features: decay support (`decay_enabled`, `decay_halflife_days`), cached aggregates
  (`cached_value`, `cached_variance`, `cached_confidence`, `cached_n`, `cached_effective_n`)
- `MeasureTarget`: separate model for working layer (aspirational values)

What the requirements spec adds that existing pcl/ doesn't have:
- `measure_type` (process/material/product/resource) classification
- `value_type` (continuous/discrete/proportion/integer)
- Explicit two-layer model (operational/working) — existing uses MeasureTarget for working
- Component M2M for calculated measures — existing uses formula `[slug]` references

**Action:** On PCL build day, read existing `pcl/models.py`, `pcl/confidence.py`,
`pcl/service.py`. Migrate the existing schema to add missing fields. Keep the
deployed confidence formula (log2 is better). Keep SynaraImmutableLog for Datapoint
(correct — datapoints should be write-once). Add the new classification fields
via Django migration on the existing tables.

**Do not recreate pcl/models.py from scratch. Extend what's deployed.**

Datapoint FK to Measure must be changed from CASCADE to PROTECT (red/blue finding #10:
CASCADE bypasses SynaraImmutableLog.delete()).

### 3.3 governance/models.py

**⚠ RED/BLUE FINDING (2026-05-04): syn/synara/ is middleware, not a Django app
(no apps.py, not in INSTALLED_APPS, settings.py explicitly says "NOT registered").
Governance models live in a new `governance/` app, not in syn/synara/.**

```python
from syn.core.base_models import SynaraEntity


class GovernanceRule(SynaraEntity):
    """
    A rule that Synara tracks and reinforces/decays based on outcomes.
    
    Rules can be system-defined (shipped), learned (from workflow patterns),
    or manual (admin-created). Confidence adjusts via Bayesian updating
    based on GovernanceDecision outcomes.
    
    REQ-SY-1, REQ-SY-8, REQ-SY-9
    """
    name = models.CharField(max_length=200)
    description = models.TextField()
    rule_type = models.CharField(
        max_length=20,
        choices=[
            ('workflow', 'Workflow'),
            ('threshold', 'Threshold'),
            ('routing', 'Routing'),
            ('validation', 'Validation'),
        ],
    )
    condition = models.JSONField(default=dict)
    action = models.JSONField(default=dict)
    confidence = models.FloatField(default=0.5)
    times_applied = models.IntegerField(default=0)
    times_good_outcome = models.IntegerField(default=0)
    times_bad_outcome = models.IntegerField(default=0)
    source = models.CharField(
        max_length=20,
        choices=[
            ('system', 'System'),
            ('learned', 'Learned'),
            ('manual', 'Manual'),
        ],
    )
    is_active = models.BooleanField(default=True)

    def update_confidence(self, outcome):
        """
        Bayesian update: good outcomes increase confidence,
        bad outcomes decrease it. Uses the same adjusted LR formula
        from core/bayesian.py.
        """
        self.times_applied += 1
        if outcome == 'good':
            self.times_good_outcome += 1
        elif outcome == 'bad':
            self.times_bad_outcome += 1
        # Simple Bayesian: confidence = good / (good + bad) with Laplace smoothing
        alpha = self.times_good_outcome + 1
        beta = self.times_bad_outcome + 1
        self.confidence = round(alpha / (alpha + beta), 4)
        self.save(update_fields=['confidence', 'times_applied',
                                  'times_good_outcome', 'times_bad_outcome'])

    class Meta:
        ordering = ['-confidence']


class GovernanceDecision(SynaraEntity):
    """
    Immutable record of a governance event: a rule or heuristic fired,
    and the user accepted, overridden, or modified it.
    
    This record is IMMUTABLE after creation. It captures the decision only.
    Outcomes are recorded as separate GovernanceOutcome records that
    reference back to the decision. This avoids the immutability paradox
    (red/blue finding #1: can't update 'pending' if save() blocks updates).
    
    REQ-SY-1 (confidence from outcomes)
    """
    rule = models.ForeignKey(
        GovernanceRule, null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='decisions',
    )
    heuristic = models.ForeignKey(
        'workflows.Heuristic', null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='decisions',
    )
    instance = models.ForeignKey(
        'workflows.WorkflowInstance', null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name='governance_decisions',
    )
    decision = models.CharField(
        max_length=20,
        choices=[
            ('accepted', 'Accepted'),
            ('overridden', 'Overridden'),
            ('modified', 'Modified'),
        ],
    )
    context = models.JSONField(default=dict)

    def save(self, *args, **kwargs):
        if self.pk and not self._state.adding:
            raise ValueError("GovernanceDecision records are immutable after creation")
        super().save(*args, **kwargs)

    class Meta:
        ordering = ['-created_at']


class GovernanceOutcome(SynaraEntity):
    """
    Records the outcome of a governance decision. Separate from the decision
    itself so both are individually immutable — the decision is a fact,
    the outcome is a later fact. Both append-only.
    
    When an outcome is recorded, it triggers GovernanceRule.update_confidence()
    to close the Bayesian learning loop.
    """
    decision = models.ForeignKey(
        GovernanceDecision,
        on_delete=models.PROTECT,
        related_name='outcomes',
    )
    outcome = models.CharField(
        max_length=20,
        choices=[
            ('good', 'Good'),
            ('bad', 'Bad'),
            ('neutral', 'Neutral'),
        ],
    )
    evidence = models.JSONField(default=dict)

    def save(self, *args, **kwargs):
        if self.pk and not self._state.adding:
            raise ValueError("GovernanceOutcome records are immutable after creation")
        super().save(*args, **kwargs)

    class Meta:
        ordering = ['-created_at']
```

### 3.4 governance/contracts.py (same app)

```python
from syn.core.base_models import SynaraEntity


class Contract(SynaraEntity):
    """
    A visible, user-severable data contract between two apps.
    Pull-only: consumer pulls from source. Source never pushes.
    
    HC-15 (immutable audit), REQ-SY-4 (visible), REQ-SY-5 (pull-only)
    FLOW-1, FLOW-3 from integration constraints.
    """
    name = models.CharField(max_length=200)
    source_app = models.CharField(max_length=50)
    source_type = models.CharField(max_length=50)
    consumer_app = models.CharField(max_length=50)
    consumer_type = models.CharField(max_length=50)
    config = models.JSONField(default=dict)
    version = models.PositiveIntegerField(default=1)
    supersedes = models.ForeignKey(
        'self', null=True, blank=True,
        on_delete=models.PROTECT,
        related_name='superseded_by',
    )
    is_active = models.BooleanField(default=True)
    severed_at = models.DateTimeField(null=True, blank=True)
    severed_by = models.CharField(max_length=200, blank=True)

    class Meta:
        ordering = ['source_app', 'consumer_app']
```

---

## 4. API Surface

### 4.1 workflows/urls.py

```
/api/workflows/defs/                    GET     List workflow definitions
/api/workflows/defs/                    POST    Create workflow definition
/api/workflows/defs/<id>/               GET     Get workflow definition detail
/api/workflows/defs/<id>/               PUT     Update workflow definition
/api/workflows/defs/<id>/fork/          POST    Fork a workflow definition

/api/workflows/instances/               GET     List instances (with filters)
/api/workflows/instances/               POST    Create instance (from def or ad-hoc)
/api/workflows/instances/<id>/          GET     Get instance with steps
/api/workflows/instances/<id>/step/     POST    Add step to instance
/api/workflows/instances/<id>/complete/ POST    Mark complete, trigger heuristic learning
/api/workflows/instances/<id>/lock/     POST    Lock for validation (REQ-IN-4)
/api/workflows/instances/<id>/export/   GET     Full JSON export (HC-5)

/api/workflows/heuristics/              GET     List heuristics (by confidence)
/api/workflows/heuristics/match/        POST    Find matching heuristics for context
/api/workflows/heuristics/<id>/         GET     Get heuristic detail
```

### 4.2 pcl/urls.py

```
/api/pcl/measures/                      GET     List measures (filter by type, layer)
/api/pcl/measures/                      POST    Create measure
/api/pcl/measures/<id>/                 GET     Get measure with latest datapoint
/api/pcl/measures/<id>/resolve/         GET     Resolve calculated measure (chain)
/api/pcl/measures/<id>/datapoints/      GET     Datapoint history
/api/pcl/measures/<id>/datapoints/      POST    Add datapoint
/api/pcl/measures/<id>/staleness/       GET     Staleness check (REQ-SC-4)
/api/pcl/export/                        GET     Full PCL export (HC-5)
```

### 4.3 Governance (added to existing syn/ API or standalone)

```
/api/governance/rules/                  GET     List active rules
/api/governance/rules/<id>/             GET     Rule detail with decision history
/api/governance/decisions/              GET     Decision log
/api/governance/decisions/              POST    Record decision (accepted/overridden)
/api/governance/decisions/<id>/outcome/ POST    Record outcome (closes learning loop)
/api/governance/contracts/              GET     List active contracts
/api/governance/contracts/<id>/sever/   POST    Sever a contract (user action)
```

---

## 5. svend Plugin (svend@eric-tools)

Built using plugin-dev@eric-tools. Dog-foods the Claude facilitation layer.

### 5.1 Skills

```
svend:cr          Create and manage ChangeRequests. Replaces manual shell.
                  Automates: create → submit → risk assess → approve → in_progress.

svend:compliance  Run compliance checks inline. Report results.
                  Wraps: python manage.py run_compliance

svend:workflow    Create, query, and propose workflows.
                  - svend:workflow list — show available workflow defs
                  - svend:workflow run <def_id> — start an instance
                  - svend:workflow log <instance_id> — show steps/provenance
                  - svend:workflow propose — search heuristics for current context

svend:pcl         Query and write PCL measures and datapoints.
                  - svend:pcl measures — list measures
                  - svend:pcl read <measure> — latest value + confidence
                  - svend:pcl write <measure> <value> — add datapoint
                  - svend:pcl stale — check for stale measures

svend:heuristic   Search heuristics, propose matches for current work.
                  Background search + toast-style proposal.

svend:clean       Lint models for SynaraEntity inheritance, check patterns.
                  Run on any model file edit.
```

### 5.2 Hooks

```yaml
hooks:
  - event: post_tool_use
    match: "Edit models.py"
    command: "check SynaraEntity inheritance on edited model"

  - event: pre_commit
    command: "verify active CR exists (wire existing check_cr.py)"

  - event: session_start
    command: "source /etc/svend/env && show active workflows and CRs"
```

### 5.3 Plugin JSON Structure

```json
{
  "name": "svend",
  "version": "0.1.0",
  "description": "SVEND container system — workflow management, PCL, governance",
  "skills": [
    { "name": "cr", "path": "skills/cr.md" },
    { "name": "compliance", "path": "skills/compliance.md" },
    { "name": "workflow", "path": "skills/workflow.md" },
    { "name": "pcl", "path": "skills/pcl.md" },
    { "name": "heuristic", "path": "skills/heuristic.md" },
    { "name": "clean", "path": "skills/clean.md" }
  ],
  "hooks": "hooks.json"
}
```

---

## 6. Pre-Built Workflow Templates (Tier 1)

Ship with the product. Source = 'system'. REQ-WF-5.

| Template | Steps | Persona Source |
|----------|-------|---------------|
| Cpk-to-PPAP | Import CMM data → Capability study → Control chart → PPAP report | Greg Linden |
| Incoming Inspection | Import data → Capability on N characteristics → Flag below Cpk 1.33 → Report | David Kwon |
| Scrap Pareto | Import scrap log → Pareto analysis → Dollar quantification → Dashboard | Greg Linden |
| CAPA Chain | Complaint → NCR → Investigation → Corrective action → Training → Verification | Priya Chakraborty |
| Check Weigher Analysis | Import CSV → Line/shift/product routing → Overfill analysis → Dollar impact | Tameka Jackson |
| Short-Run SPC | Import data → DNOM/Q-chart for small lots → Control limits → Report | Ray Nguyen |
| Basic Capability | Import data → Normality check → Cpk/Ppk → Report with reasoning trail | General |

Each template is a WorkflowDef with source='system', visibility='public', tier='free'.

---

## 7. Integration with Existing System

### 7.1 Analysis Router → Workflow Logging

The existing `analysis/router.py` dispatches analyses. Wire it to log each dispatch as a WorkflowStep:

```python
# In analysis/router.py dispatch function (or wrapper)
def dispatch_with_logging(analysis_type, analysis_id, inputs, user, instance=None):
    """Wrap existing dispatch to log as workflow step."""
    result = registry.dispatch(analysis_type, analysis_id, inputs)
    
    if instance:
        WorkflowStep.objects.create(
            instance=instance,
            sequence=instance.steps.count() + 1,
            action=f'{analysis_type}.{analysis_id}',
            inputs=inputs,
            outputs=result,
            tool=analysis_type,
            status='completed',
            created_by=user.email,
        )
    
    return result
```

This is the bridge. Existing analyses work exactly as before. When executed within a workflow context, they produce provenance.

### 7.2 QMS Templates → Workflow Templates

QMS ToolTemplates (a3-report, ishikawa, fmea, rca, etc.) can be wrapped as WorkflowDefs:

```python
# One-time migration: for each QMS ToolTemplate, create a WorkflowDef
WorkflowDef.objects.create(
    name=f'QMS: {template.name}',
    schema={'qms_template_id': str(template.id), 'sections': template.schema},
    source='system',
    visibility='public',
    tier='free',  # or 'addon' for advanced versions
    tags=['qms', template.name],
)
```

### 7.3 PCL ← Existing Tools

Existing tools become PCL writers via workflow steps:

| Tool | What it writes to PCL | source_type |
|------|----------------------|-------------|
| SPC engine | Control limits, Cpk/Ppk values | automated |
| DOE / experimenter | Factor effects, optimal settings | workbench |
| Capability analysis | Process indices | workbench |
| Manual entry form | User-entered values | manual |
| VSM (when PCL-bound) | Step characteristics | manual |

### 7.4 Contracts

Initial contract wiring (pull-only, visible, user-severable):

| Source | Consumer | Contract |
|--------|----------|----------|
| Workbench (analysis/) | Workflows | Analysis results → workflow step outputs |
| SPC | PCL | Control limits, capability indices → measures |
| PCL | Workflows | Current measure values → workflow context |
| Workflows | Governance | Instance outcomes → rule confidence updates |
| Governance | Workflows | Heuristic proposals → suggested next steps |

---

## 8. Export Architecture (HC-5, HC-10)

Every app provides full structured export from day one, any tier.

```python
# workflows/views.py
@require_auth
def export_instance(request, instance_id):
    """Full JSON export of workflow instance with all steps and provenance."""
    instance = WorkflowInstance.objects.get(id=instance_id, tenant_id=request.tenant_id)
    return JsonResponse({
        'workflow_def': serialize(instance.workflow_def),
        'instance': serialize(instance),
        'steps': [serialize(s) for s in instance.steps.all()],
        'governance_decisions': [serialize(d) for d in instance.governance_decisions.all()],
        'exported_at': now().isoformat(),
        'format_version': '1.0',
    })

# pcl/views.py
@require_auth
def export_pcl(request):
    """Full JSON + CSV export of all measures and datapoints."""
    measures = Measure.objects.for_tenant(request.tenant_id)
    return JsonResponse({
        'measures': [serialize(m, include_datapoints=True) for m in measures],
        'exported_at': now().isoformat(),
        'format_version': '1.0',
    })
```

CSV export for PCL datapoints also available (for Minitab interop — HC-12).

---

## 9. Compliance & Testing (Control Phase)

### 9.1 New Compliance Checks

Added to `syn/audit/compliance.py` registry:

| Check | Category | Schedule | What it verifies |
|-------|----------|----------|-----------------|
| workflow_integrity | processing_integrity | Daily | All completed instances have >= 1 step. No orphaned steps. |
| pcl_staleness | processing_integrity | Daily | Flag measures with no datapoint in > 30 days. |
| governance_confidence | processing_integrity | Weekly | Flag rules with confidence < 0.3 (decaying). |
| contract_health | processing_integrity | Daily | All active contracts have both source and consumer apps installed. |
| export_availability | availability | Weekly | Export endpoints return valid JSON for sample data. |
| synaraentity_compliance | processing_integrity | Weekly | All models in new apps inherit SynaraEntity. |

### 9.2 Golden File Testing

Minitab-matched outputs for statistical analyses routed through workflows (HC-17):

```python
# workflows/tests/test_golden_files.py
class TestGoldenFiles(TestCase):
    """
    Known-answer tests. Each golden file contains:
    - Input data
    - Expected Minitab output
    - Tolerance (for floating point)
    
    REQ-WF-8 (byte-for-byte reproducibility)
    HC-17 (numbers match Minitab)
    """
    fixtures_dir = Path(__file__).parent / 'golden_files'
    
    def test_capability_matches_minitab(self):
        golden = json.load(open(self.fixtures_dir / 'capability_normal.json'))
        result = run_workflow_template('basic_capability', golden['input'])
        assert_close(result['cpk'], golden['expected']['cpk'], atol=0.001)
        assert_close(result['ppk'], golden['expected']['ppk'], atol=0.001)
```

### 9.3 Calibration (CAL-001)

- Statistical outputs verified against known datasets
- Coverage ratchet: new analyses must have golden file tests before merge
- Forge packages provide synthetic datasets with known properties for calibration

---

## 10. Build Sequence (11 days)

Infrastructure exists (SynaraEntity, audit, compliance, 200+ analyses). This is
configuration and wiring, not greenfield. Models inherit SynaraEntity — migrations
and domain fields. CHG-001 process overhead absorbed into daily work.

### Phase 1: Build the Container (~5 days)

```
Day 1: Scaffold + Workflow Models
  - Create workflows/ and governance/ apps (models, admin, urls, views, tests/)
  - Add to INSTALLED_APPS, makemigrations, migrate
  - WorkflowDef, WorkflowInstance, WorkflowStep (PROTECT not CASCADE), Heuristic
  - Scaffold svend@eric-tools plugin via plugin-dev
  - CR for scaffolding

Day 2: Governance Models
  - GovernanceRule, GovernanceDecision (immutable), GovernanceOutcome (separate, also immutable)
  - Contract model (versioned, pull-only, string app refs with choices validation)
  - makemigrations, migrate

Day 3: PCL Diff/Merge
  - Read existing ~/kjerne/pcl/ models, confidence, service
  - Diff with requirements spec — add missing fields (measure_type, value_type)
  - Keep deployed confidence formula (log2), keep SynaraImmutableLog for Datapoint
  - Change Datapoint FK to Measure from CASCADE to PROTECT
  - Migration on existing tables, NOT recreate

Day 4: Engine + API
  - Workflow engine (capture, retain, heuristic search, promote ad-hoc → def)
  - Views + urls for workflows/ and governance/
  - Export endpoints (JSON + CSV, HC-5)
  - Router wrapping: try/except, logging failure NEVER blocks analysis result
  - Feature flags per app for incremental activation

Day 5: Frontend Decision + Plugin
  - Decide on frontend approach (container view, top/bottom split)
  - Wire analysis/router.py dispatch → workflow step logging
  - Wire SPC output → PCL Datapoint creation
  - svend plugin: cr, workflow, pcl skills
```

### Phase 2: Testing the Container (~4 days)

```
Day 5: Model + Engine Tests
  - TDD: WorkflowDef/Instance/Step lifecycle
  - TDD: Heuristic matching + confidence update
  - TDD: GovernanceDecision immutability
  - TDD: Datapoint confidence computation (edge cases: n=0, n=1, n=10000)
  - TDD: Calculated measure resolution + staleness

Day 6: Integration Tests
  - Full workflow: data import → analysis → workflow step → PCL write → export
  - Governance loop: rule fires → decision recorded → outcome → confidence update
  - Contract wiring: source → consumer pull
  - Idempotency: crash mid-step, resume correctly

Day 7: Golden Files + Compliance
  - Golden file tests for Minitab-matched outputs (HC-17)
  - Add 6 new compliance checks to syn/audit registry
  - Run full compliance suite, fix failures
  - SynaraEntity inheritance audit

Day 8: Calibration + Edge Cases
  - Statistical calibration against known datasets (CAL-001)
  - Edge cases: empty data, invalid parameters, missing measures
  - Forge synthetic data for workflow templates
  - Max depth guard tested (recursive workflow prevention)
```

### Phase 3: Integration (~3 days)

```
Day 9: App Integration
  - QMS template → WorkflowDef migration script
  - Pre-built workflow templates (7 system templates)
  - Test each template end-to-end with sample data

Day 10: Plugin + Hooks
  - svend plugin: heuristic, clean, compliance skills
  - Hooks: post-model-edit SynaraEntity check, session-start context
  - Test plugin end-to-end

Day 11: Standardize + Tag + Screen Architecture
  - Document patterns (WORKFLOW-001, PCL-001)
  - Update ARCHITECTURE.md, INTEGRATION_SPEC.md
  - Screen architecture: single-pane with home base, pin panel, state preservation
  - Tag v0.1.0 of container system
```

### Post-Conference Improvements (folded into build)

From S2 code quality review (2026-05-04):
- ✅ GovernanceDecision: immutable (save() prevents update after creation)
- ✅ WorkflowStep: idempotency_key for crash recovery
- ✅ Datapoint confidence: floor at 0.01, guard against n=0
- ✅ Contract: versioned with supersedes FK
- ✅ WorkflowInstance: MAX_DEPTH=3, parent_step FK
- ✅ Feature flags per app for incremental activation
- GovernanceRule source='learned' deferred until governance can govern itself

---

## 11. Phase-Out Plan (Existing Apps)

Not in 30-day scope. Documented for future.

| Legacy | Replaced By | Phase-Out Trigger |
|--------|------------|-------------------|
| agents_api analysis dispatch | workflows/ + analysis/router | When 80% of analyses run through workflow instances |
| agents_api QMS models (FMEA, RCA, A3) | qms/ ToolTemplates wrapped as WorkflowDefs | When QMS template picker UI ships |
| agents_api Board | Standalone — no replacement needed yet | When workflow-based collaboration ships |
| core/synara.py (belief engine) | syn/synara/governance.py | When governance confidence tracking is validated |
| prova/ | Workflow retention + governance heuristics + Claude facilitation | PROVA is replaced. Claude absorbs the facilitator role. Workflow retention + Synara governance replace the knowledge graph. |

---

## 12. Resolved Decisions (2026-05-04)

### 12.1 One Workflow Engine

There is one workflow engine: WorkflowDef → WorkflowInstance → WorkflowStep → Heuristic. The QMS WorkflowTemplate/Phase/Transition system is absorbed or retired. Phase gates, approval gates, CFR Part 11 electronic signatures — all extensible via marketplace add-ins on the one engine. The engine is clean; the marketplace is how it grows capabilities.

### 12.2 PCL is Hybrid Update

PCL is deployed at `~/kjerne/pcl/` with migration 0001_initial. Three models already exist with log2 confidence curves, decay support, cached aggregates, formula evaluator, and service layer. The spec's PCL work is diff/merge — add missing classification fields (`measure_type`, `value_type`) to existing schema. Change Datapoint FK from CASCADE to PROTECT. Do not recreate.

### 12.3 Single-Pane Layout Validated (4/4 Focus Group)

Single top/bottom split validated by all four personas (Greg Linden, Marcus Wade, Priya Chakraborty, James Okafor). Multi-pane is compensation for tools that don't cross-reference, not an analytical requirement.

Required features that make single-pane viable:
- **Home base:** Default view per user context (floor: SPC chart; office: last tool). Gravity well — nothing displaces without explicit action.
- **Pin panel:** Advisor panel slot can pin reference data (CMM output, spec sheet, linked record) alongside working tool. Same slot as Claude's slide-in. Required for QE adoption.
- **State preservation:** Switching tools preserves full state (scroll, zoom, annotations, form data). Universal requirement.
- **Snapshot/freeze:** Save view state for later comparison. Power user need.
- **Sub-second switching:** Pane transitions must feel instant.
- **Thin command strip on floor:** Minimal bar on tablets, expands on tap.

See: `docs/superpowers/innovate/2026-05-04-single-pane-layout-focus.md`

### 12.4 Separation of Concerns — The Trust Architecture (morph + conference, unanimous)

**PCL** = single system of record (what we know).
- Process characteristics with Bayesian confidence and provenance
- Rules layer for relational knowledge: heuristic (where to go, what to fetch, temporal context) + governance on interaction + knowledge compression
- NOT just flat measures — also multi-factor relationships that survive context windows

**Synara** = governance only (what's allowed).
- Contracts, audit trail, reversibility gate, approvals
- Does NOT store knowledge. Does NOT accumulate rules. Not middleware.
- Governs writes, not reads. Governs actions, not recommendations.

**Claude** = facilitator, assembles own context.
- Reads PCL directly (no intermediary, no session-assembled context)
- Recommendations ungoverned. Actions governed via Synara write contracts.
- `context_cited` on every write from Day 1 — snapshot of PCL entries referenced. Uses existing SynaraImmutableLog UUID hash chains.

**Workflows** = deterministic programs. The hard layer that carries the product.
- Explicit: user authors a workflow — tools, data bindings, triggers (schedule, PCL update, threshold breach), conditionals (staleness check: "only if PCL fields updated since last run"), governance rules (Synara approval gates)
- Named, shareable, schedulable. These are just programs. They always work.
- No drag-and-drop designer — authored via session save or direct configuration
- Staleness check prevents the most common manufacturing antipattern: running reports on stale data because it's Tuesday

**Sessions** = workspaces (the atomic unit). One session = one task.
- Tool + Data Bindings (PCL) + Layout + Context = Workspace
- User-defined open/close boundary (a task, not a time window)
- Every session logged immutably by Synara: tools, configs, schemas, PCL reads/writes, outcomes, temporal context
- "Save as workflow" = strip chat/exploration, keep tool configs + data bindings + layout, leave input ports open
- Ships in 11-day build, not deferred

**Heuristics** = soft convenience layer on top. NOT load-bearing.
- Pattern matching on session logs (schema shape + user + time + role) for suggestions only
- "Looks like your Thursday report" — dismiss or accept, no consequence either way
- Bayesian reinforcement from outcomes. Fires only above confidence threshold.
- Nice to have, not the product. 10% of value, not 90%.

**Access pattern:** Reads free. Writes through Synara contracts, gated by binary reversibility (reversible or irreversible, no gray area).

**Learning loop:** Outcomes → PCL Bayesian update. No separate learning engine.

**Governance:** Outcomes-only. Enterprise governed-recommendation mode deferred until regulated customer demands it.

See: `docs/superpowers/innovate/2026-05-04-separation-of-concerns-morph.md`
See: `docs/superpowers/innovate/2026-05-04-separation-of-concerns-conference.md`

### 12.5 Workflows — Hard Layer and Soft Layer (wargame + correction, 2026-05-04)

**The hard layer carries the product. The soft layer is convenience.**

**Hard layer (deterministic, 90% of value):**
- Explicit workflows: authored by users, deterministic execution. "Pull PCL fields A,B,C → check staleness → run capability → generate report → require sign-off if Cpk < 1.33." This is a program. It always works.
- Session save: user completes a task, saves it as a workflow. System strips chat/exploration, keeps tool configs + data bindings + layout, leaves input ports open for next run. Simple save, not promotion from a heuristic.
- Triggers: schedule (cron), PCL data update, threshold breach. Conditionals: staleness check ("only if updated since last run"). Governance: Synara approval gates on irreversible actions.
- Named, shareable, schedulable. Lisa's Thursday report is an explicit workflow she authored once.
- No drag-and-drop workflow designer. Configuration, not visual programming.

**Soft layer (heuristic, 10% of value — convenience only):**
- Synara logs every session immutably. Heuristic engine matches patterns across logs.
- Schema matching (column types, ranges, structure), temporal patterns (user + day + frequency), role patterns (operators with this data shape usually need SPC)
- Bayesian reinforcement: accepted + good outcome → confidence up; dismissed → confidence down; decay if unused
- Surfacing: Claude says "Looks like your Thursday report" / quick-action bar shows suggestion above confidence threshold
- User can dismiss with no consequence. Wrong suggestions weaken the heuristic automatically.
- **Not load-bearing.** If the heuristic engine broke tomorrow, every explicit workflow still runs. Users just don't get suggestions.

**The atomic unit is the workspace, not the tool:**
- Tool + Data Bindings + Layout + Context = Workspace
- A session IS a workspace (one task, user-defined open/close)
- A saved workflow IS a workspace template (input ports open, everything else frozen)
- Users live in workspaces. Search + docs for when they need something new.

**Governance is practitioner knowledge captured at point of use:**
- Lisa rejects a tool → that rejection + reasoning enters PCL
- Next site's QE sees "flagged by QE, methodological concern" — doesn't re-discover it
- No approval meetings. Knowledge built in the trenches.

See: `docs/superpowers/innovate/2026-05-04-single-pane-navigation-wargame.md`

### 12.6 Navigation Architecture (wargame validated, 2026-05-04)

**Without Claude, navigation is: workspace loads → search bar → documentation.**

Wargame finding: users abandon categories after 2 sessions, live in favorites/search, and build personal micro-products of 2-8 tools. The sidebar is onboarding infrastructure, not daily navigation.

**Three modes for three personas:**
1. **Operators (Marcus):** Destination mode. Pinned workspace loads on open. Zero navigation. One screen, pre-configured by someone else (Lisa, or implicit heuristic from role pattern).
2. **Engineers (Lisa):** Workbench mode. Personal workspace with favorited tools, data bus connecting everything, layout she controls. Search for new tools when needed.
3. **New users:** Guided path mode. Role-based onboarding pre-configures starter workspace. Contextual suggestions after analysis completion. Inline prompts ("Import data" inside the Capability tool, not as a separate step).

**What the UI is NOT:**
- Not an app launcher (kills operator speed)
- Not a component palette (too abstract for new users)
- Not separated by apps OR by components — it's workspace-first, tools are inside the workspace
- Not a sidebar you browse daily (sidebar is for onboarding and expansion only)

**The data bus is the moat.** What makes this a platform, not a tool collection: imported data is visible to all tools via shared PCL bindings. No re-upload, no export/import between tools, no manual wiring. The bus is invisible infrastructure that users feel but don't see.

**Two-mode architecture (focus group validated):**
- Power-user desktop: full workspace, multiple tools, pin panel, command strip
- Operator kiosk: single pinned view, minimal chrome, thin command strip on tablets

### 12.7 Marketplace as Extension Mechanism

Claude is in the product. The workflow engine is what Claude orchestrates through. PCL is what Claude reads. Synara governance provides confidence scores. The marketplace extends Claude's facilitation surface — install an add-in, Claude can facilitate through its capabilities. The engine stays clean.

---

## 13. Requirements Traceability

Every requirement from `3P_REQUIREMENTS_SPEC.md` mapped to this design:

| Requirement | Design Location |
|-------------|----------------|
| REQ-WF-1 through REQ-WF-8 | Section 3.1 (WorkflowDef, WorkflowInstance, WorkflowStep) |
| REQ-IN-1 through REQ-IN-6 | Section 3.1 (WorkflowInstance, WorkflowStep, is_locked) |
| REQ-SY-1 through REQ-SY-9 | Section 3.3 (GovernanceRule, GovernanceDecision, Contract) |
| REQ-PCL-1 through REQ-PCL-10 | Section 3.2 (Measure, Datapoint, confidence.py, resolution.py) |
| REQ-CL-1 through REQ-CL-12 | Section 5 (svend plugin — prototype of facilitation layer) |
| REQ-SC-1 through REQ-SC-15 | Deferred to UI design phase (data model supports all) |
| HC-1 through HC-24 | Section 4 (API design), Section 8 (export), Section 9 (compliance) |
| MR-1 through MR-6 | Section 7 (integration), Section 3.3 (contracts) |

**Screen architecture resolved:** REQ-SC-1 through REQ-SC-3f validated via focus group (2026-05-04). REQ-SC-4 through REQ-SC-15 deferred to UI implementation — data models support all of these.

**Deferred to marketplace phase:** Tier 2 add-ins, community contribution workflow, marketplace discovery UI.

---

*Next step: Implementation plan via writing-plans skill.*
