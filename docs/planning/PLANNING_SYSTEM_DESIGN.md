# Planning System Design — CI/CD-Style Project Management

**ID:** `plan-system-001`
**Version:** 1.0
**Date:** 2026-03-04
**Authors:** Eric + Claude (Systems Architect)
**Status:** DRAFT — Architecture Design
**Related:** `docs/planning/NEXT_GEN_QMS_MASTER_PLAN.md`, RDM-001, CHG-001

---

## 1. Problem Statement

Today, Svend has three disconnected planning layers:

```
Strategic (docs/planning/*.md)     ← ad-hoc markdown, no structure, not queryable
Public    (RoadmapItem)            ← flat, quarter-based, no hierarchy
Execution (ChangeRequest)          ← per-change, no upward link to "why"
```

**What's missing:**
- No way to say "FEAT-042 (Notification system) is 40% complete — 2 of 5 tasks done"
- No way to say "Phase 3 is blocked because FEAT-039 depends on FEAT-041"
- No way to hand Claude `FEAT-042` in a new session and have it reconstruct full context
- No dependency graph between features
- `PlanDocument` is just a markdown blob — no structured tracking
- Feature IDs like `E3-001` in the master plan are prose, not database records

**What we want:**
A structured, queryable, UUID-tracked hierarchy where every planned capability is a grepable record with dependencies, status rollup, and bidirectional links to standards, ChangeRequests, and roadmap items.

---

## 2. Design Principles

1. **Three-tier hierarchy:** Initiative → Feature → Task
2. **Dual identifiers:** UUID PK (database) + short human ID (grepable)
3. **Short IDs are sacred:** Once assigned, never reused, always visible
4. **Status rolls up:** Initiative progress computed from child features
5. **Dependencies are first-class:** Features form a DAG, not just a list
6. **CLI-queryable:** Management command lets Claude reconstruct context in any session
7. **Standards-linked:** Features reference the standards they implement or require
8. **Bidirectional:** Feature ↔ ChangeRequest ↔ RoadmapItem ↔ Initiative

---

## 3. Model Hierarchy

```
Initiative (strategic theme / phase)
  ├── Feature (deliverable capability)
  │     ├── Task (implementation work item)
  │     │     └── ChangeRequest (execution record)
  │     ├── Task
  │     │     └── ChangeRequest
  │     └── dependencies → [Feature, Feature, ...]
  ├── Feature
  │     └── ...
  └── RoadmapItem (public-facing)

Short IDs:
  Initiative:  INIT-001, INIT-002, ...
  Feature:     FEAT-001, FEAT-002, ...
  Task:        TASK-001, TASK-002, ...
```

### 3.1 Initiative

A strategic theme or phase. Maps to sections of the master plan (e.g., "Phase 3: Enterprise Foundation").

```python
class Initiative(models.Model):
    """
    Strategic planning initiative — a themed phase of work.

    Maps to phases in the master plan. Contains features.
    Short ID: INIT-<seq> (e.g., INIT-003).
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    short_id = models.CharField(
        max_length=10, unique=True, db_index=True, editable=False,
        help_text="Human-readable ID: INIT-001, INIT-002, ..."
    )

    class Status(models.TextChoices):
        PLANNED = "planned", "Planned"
        ACTIVE = "active", "Active"
        COMPLETED = "completed", "Completed"
        ON_HOLD = "on_hold", "On Hold"

    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    status = models.CharField(
        max_length=15, choices=Status.choices, default=Status.PLANNED, db_index=True
    )
    target_quarter = models.CharField(
        max_length=7, blank=True,
        help_text="Target quarter: Q1-2026, Q2-2026, etc."
    )
    sort_order = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "planning_initiatives"
        ordering = ["sort_order", "-created_at"]

    def save(self, *args, **kwargs):
        if not self.short_id:
            last = Initiative.objects.order_by("-short_id").first()
            seq = 1
            if last and last.short_id.startswith("INIT-"):
                seq = int(last.short_id.split("-")[1]) + 1
            self.short_id = f"INIT-{seq:03d}"
        super().save(*args, **kwargs)

    def __str__(self):
        return f"[{self.short_id}] {self.title} ({self.status})"

    @property
    def progress(self):
        """Compute progress from child features."""
        features = self.features.all()
        if not features:
            return 0
        completed = features.filter(status="completed").count()
        return round(completed / features.count() * 100)
```

### 3.2 Feature

A deliverable capability. The atomic unit of planning — something a user can see or a standard requires.

```python
class Feature(models.Model):
    """
    A deliverable capability tracked through planning → implementation.

    Short ID: FEAT-<seq> (e.g., FEAT-042).
    Links to initiatives, standards, ISO clauses, and ChangeRequests.
    Supports dependency DAG between features.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    short_id = models.CharField(
        max_length=10, unique=True, db_index=True, editable=False,
        help_text="Human-readable ID: FEAT-001, FEAT-002, ..."
    )

    class Status(models.TextChoices):
        BACKLOG = "backlog", "Backlog"
        PLANNED = "planned", "Planned"
        IN_PROGRESS = "in_progress", "In Progress"
        BLOCKED = "blocked", "Blocked"
        COMPLETED = "completed", "Completed"
        DEFERRED = "deferred", "Deferred"
        CANCELLED = "cancelled", "Cancelled"

    class Priority(models.TextChoices):
        CRITICAL = "critical", "Critical"
        HIGH = "high", "High"
        MEDIUM = "medium", "Medium"
        LOW = "low", "Low"

    # Identity
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    acceptance_criteria = models.TextField(
        blank=True,
        help_text="What must be true for this feature to be complete"
    )

    # Classification
    initiative = models.ForeignKey(
        Initiative, on_delete=models.CASCADE, related_name="features"
    )
    status = models.CharField(
        max_length=15, choices=Status.choices, default=Status.BACKLOG, db_index=True
    )
    priority = models.CharField(
        max_length=10, choices=Priority.choices, default=Priority.MEDIUM, db_index=True
    )

    # Standards & compliance linkage
    iso_clause = models.CharField(
        max_length=20, blank=True, db_index=True,
        help_text="ISO 9001 clause reference (e.g., §7.5, §10.2)"
    )
    standards = models.JSONField(
        default=list, blank=True,
        help_text="Standards this feature implements or requires: ['SIG-001', 'QMS-001']"
    )

    # Dependencies (DAG)
    depends_on = models.ManyToManyField(
        "self", symmetrical=False, related_name="blocks", blank=True,
        help_text="Features that must complete before this one can start"
    )

    # Cross-references (UUID links, Synara convention)
    roadmap_item_id = models.UUIDField(
        null=True, blank=True, db_index=True,
        help_text="Public RoadmapItem UUID (if customer-visible)"
    )
    change_request_ids = models.JSONField(
        default=list, blank=True,
        help_text="ChangeRequest UUIDs linked to this feature"
    )
    legacy_id = models.CharField(
        max_length=20, blank=True, db_index=True,
        help_text="ID from master plan (e.g., E3-001) — migration reference"
    )

    # Timestamps
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "planning_features"
        ordering = ["initiative", "priority", "-created_at"]
        indexes = [
            models.Index(fields=["initiative", "status"]),
            models.Index(fields=["iso_clause"]),
        ]

    def save(self, *args, **kwargs):
        if not self.short_id:
            last = Feature.objects.order_by("-short_id").first()
            seq = 1
            if last and last.short_id.startswith("FEAT-"):
                seq = int(last.short_id.split("-")[1]) + 1
            self.short_id = f"FEAT-{seq:03d}"
        super().save(*args, **kwargs)

    def __str__(self):
        return f"[{self.short_id}] {self.title} ({self.status})"

    @property
    def progress(self):
        """Compute progress from child tasks."""
        tasks = self.tasks.all()
        if not tasks:
            return 0
        completed = tasks.filter(status="completed").count()
        return round(completed / tasks.count() * 100)

    @property
    def is_blocked(self):
        """True if any dependency is not completed."""
        return self.depends_on.exclude(status="completed").exists()
```

### 3.3 Task

An implementation work item. Maps 1:1 to a ChangeRequest (or is the pre-cursor to creating one).

```python
class Task(models.Model):
    """
    Implementation work item within a feature.

    Short ID: TASK-<seq> (e.g., TASK-117).
    When work begins, a ChangeRequest is created and linked.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    short_id = models.CharField(
        max_length=10, unique=True, db_index=True, editable=False,
        help_text="Human-readable ID: TASK-001, TASK-002, ..."
    )

    class Status(models.TextChoices):
        TODO = "todo", "To Do"
        IN_PROGRESS = "in_progress", "In Progress"
        IN_REVIEW = "in_review", "In Review"
        COMPLETED = "completed", "Completed"
        CANCELLED = "cancelled", "Cancelled"

    class TaskType(models.TextChoices):
        MODEL = "model", "Model/Migration"
        API = "api", "API Endpoint"
        VIEW = "view", "View/Template"
        STANDARD = "standard", "Standard (documentation)"
        TEST = "test", "Test"
        INTEGRATION = "integration", "Integration"

    # Identity
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)

    # Classification
    feature = models.ForeignKey(
        Feature, on_delete=models.CASCADE, related_name="tasks"
    )
    status = models.CharField(
        max_length=15, choices=Status.choices, default=Status.TODO, db_index=True
    )
    task_type = models.CharField(
        max_length=15, choices=TaskType.choices, default=TaskType.MODEL, db_index=True
    )
    sort_order = models.IntegerField(default=0)

    # Execution link
    change_request_id = models.UUIDField(
        null=True, blank=True, db_index=True,
        help_text="ChangeRequest UUID (created when work begins)"
    )

    # Timestamps
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "planning_tasks"
        ordering = ["feature", "sort_order", "-created_at"]
        indexes = [
            models.Index(fields=["feature", "status"]),
        ]

    def save(self, *args, **kwargs):
        if not self.short_id:
            last = Task.objects.order_by("-short_id").first()
            seq = 1
            if last and last.short_id.startswith("TASK-"):
                seq = int(last.short_id.split("-")[1]) + 1
            self.short_id = f"TASK-{seq:03d}"
        super().save(*args, **kwargs)

    def __str__(self):
        return f"[{self.short_id}] {self.title} ({self.status})"
```

---

## 4. The UUID Chain — Full Traceability

Every planning artifact connects bidirectionally:

```
Initiative (INIT-003)
  │ .features →
  Feature (FEAT-042)
    │ .tasks →
    Task (TASK-117)
      │ .change_request_id →
      ChangeRequest (uuid: a3b4c5d6-...)
        │ .commit_shas →
        git commits
        │ .log_md_ref →
        log.md entry
    │ .roadmap_item_id →
    RoadmapItem (uuid: e7f8g9h0-...)
      │ → /roadmap/ (public page)
    │ .depends_on →
    Feature (FEAT-039)
    │ .standards →
    ["SIG-001", "QMS-001"]
```

**Starting from any node, you can traverse to any other:**
- "What features are in Phase 3?" → `Initiative(short_id="INIT-003").features.all()`
- "What tasks remain for notifications?" → `Feature(short_id="FEAT-042").tasks.exclude(status="completed")`
- "What ChangeRequests relate to this feature?" → `Feature.change_request_ids` → `ChangeRequest.objects.filter(id__in=...)`
- "What standard does this implement?" → `Feature.standards` + `Feature.iso_clause`
- "Is this feature blocked?" → `Feature.is_blocked` → checks all `depends_on` status

---

## 5. Short ID Convention

Short IDs are the **primary interface** for human communication and cross-session context.

| Model | Format | Example | Regex |
|-------|--------|---------|-------|
| Initiative | `INIT-<3-digit seq>` | `INIT-003` | `INIT-\d{3}` |
| Feature | `FEAT-<3-digit seq>` | `FEAT-042` | `FEAT-\d{3}` |
| Task | `TASK-<3-digit seq>` | `TASK-117` | `TASK-\d{3}` |

**Rules:**
1. Auto-generated on first save — never manually assigned
2. Never reused — if FEAT-042 is cancelled, 042 is retired
3. Globally unique per type (no scoping to initiative or feature)
4. Always displayed alongside UUID in all interfaces
5. Grepable: `grep -r "FEAT-042"` finds every reference in docs, standards, code, CLAUDE.md

**Why 3 digits?** Enough for 999 items per type. If we ever hit that, extend to 4 digits — all existing IDs remain valid.

---

## 6. Management Command — `plan`

The key interface for Claude across sessions. When the user hands Claude `FEAT-042`, this command reconstructs full context.

```
Usage:
  python manage.py plan list [--type init|feat|task] [--status STATUS] [--initiative INIT-xxx]
  python manage.py plan show FEAT-042
  python manage.py plan tree [INIT-003]
  python manage.py plan search "notification"
  python manage.py plan deps FEAT-042
  python manage.py plan blocked
  python manage.py plan progress [INIT-003]
  python manage.py plan import-master-plan
```

### 6.1 `plan list`

Lists items with optional filters:

```
$ python manage.py plan list --type feat --status in_progress

FEAT-042  Notification system (bell + SSE)           in_progress  INIT-003  P:critical
FEAT-043  Email notification with ActionToken        in_progress  INIT-003  P:critical
FEAT-051  CAPA as standalone model                   in_progress  INIT-003  P:critical

3 features (in_progress)
```

### 6.2 `plan show`

Full context dump — everything Claude needs in a new session:

```
$ python manage.py plan show FEAT-042

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEAT-042: Notification system (bell icon + SSE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

UUID:        a3b4c5d6-e7f8-9012-3456-7890abcdef01
Status:      in_progress
Priority:    critical
Initiative:  INIT-003 — Phase 3: Enterprise Foundation
ISO Clause:  Cross-cutting
Standards:   SEC-001 (SSE endpoint), QMS-001 (notification triggers)
Legacy ID:   E3-001

Description:
  Real-time notification system with bell icon and Server-Sent Events.
  Notifications for: assignment changes, approval requests, overdue items,
  SPC alarms, CAPA status changes.

Acceptance Criteria:
  - Notification model with type, recipient, read/unread, entity reference
  - SSE endpoint at /api/notifications/stream/
  - Bell icon in navbar with unread count badge
  - Mark-as-read on click, mark-all-read button
  - Notification preferences per user (which types to receive)

Dependencies:
  └── (none — this is a foundational feature)

Blocks:
  ├── FEAT-043  Email notification with ActionToken (depends on notification model)
  ├── FEAT-055  SPC alarm → auto-NCR (depends on notification delivery)
  └── FEAT-061  Management review auto-populate (depends on notification for alerts)

Tasks (3/5 completed):
  ✓ TASK-117  Create Notification model + migration         completed   CR: abc123
  ✓ TASK-118  SSE endpoint (/api/notifications/stream/)     completed   CR: def456
  ✓ TASK-119  Bell icon component + navbar integration      completed   CR: ghi789
  ○ TASK-120  Notification preferences (user settings)      in_progress CR: jkl012
  · TASK-121  Notification → email bridge (hand off to FEAT-043)  todo

Progress: 60%

Roadmap Item: e7f8g9h0-... (public: yes, shipped: no)

Change Requests:
  abc123  Create Notification model           completed   2026-03-10
  def456  SSE endpoint implementation         completed   2026-03-12
  ghi789  Bell icon frontend                  completed   2026-03-14
  jkl012  User notification preferences       in_progress 2026-03-15

Created:  2026-03-04
Updated:  2026-03-15
```

### 6.3 `plan tree`

Visual hierarchy for an initiative or the entire system:

```
$ python manage.py plan tree INIT-003

INIT-003: Phase 3 — Enterprise Foundation  [active, 25%]
├── FEAT-042  Notification system (bell + SSE)              in_progress  60%  critical
│   ├── TASK-117  Notification model + migration            ✓ completed
│   ├── TASK-118  SSE endpoint                              ✓ completed
│   ├── TASK-119  Bell icon component                       ✓ completed
│   ├── TASK-120  Notification preferences                  ○ in_progress
│   └── TASK-121  Notification → email bridge               · todo
├── FEAT-043  Email notification (ActionToken)              planned      0%   critical
│   └── ⚠ blocked by FEAT-042
├── FEAT-044  Electronic signature (CFR Part 11)            planned      0%   critical
├── FEAT-045  CAPA standalone model                         in_progress  33%  critical
│   ├── TASK-122  CAPA model + migration                    ○ in_progress
│   ├── TASK-123  CAPA lifecycle API                        · todo
│   └── TASK-124  CAPA → RCA bridge                         · todo
├── ...
└── FEAT-053  Recurrence detection                          backlog      0%   medium

12 features: 0 completed, 2 in_progress, 8 planned, 1 blocked, 1 backlog
```

### 6.4 `plan deps`

Dependency graph for a feature:

```
$ python manage.py plan deps FEAT-055

FEAT-055: SPC alarm → auto-NCR creation
  depends_on:
    └── FEAT-042  Notification system     (in_progress — 60%)
    └── FEAT-045  CAPA standalone model   (in_progress — 33%)
  blocks:
    └── FEAT-058  AI root cause suggestion (backlog)
    └── FEAT-060  Automated trending       (backlog)

Status: BLOCKED — 2 dependencies incomplete
Earliest start: when FEAT-042 and FEAT-045 complete
```

### 6.5 `plan blocked`

Lists all blocked features and their blockers:

```
$ python manage.py plan blocked

FEAT-043  Email notification         blocked by: FEAT-042 (60%)
FEAT-055  SPC alarm → auto-NCR      blocked by: FEAT-042 (60%), FEAT-045 (33%)
FEAT-058  AI root cause suggestion   blocked by: FEAT-055 (blocked)

3 features blocked
```

### 6.6 `plan progress`

Initiative-level progress summary:

```
$ python manage.py plan progress

INIT-003  Phase 3: Enterprise Foundation    active    25%  [████░░░░░░░░░░░░]  3/12 features
INIT-004  Phase 4: Doc Control & Supplier   planned    0%  [░░░░░░░░░░░░░░░░]  0/12 features
INIT-005  Phase 5: Training, Audit, Design  planned    0%  [░░░░░░░░░░░░░░░░]  0/14 features
INIT-006  Phase 6: Intelligence Layer       planned    0%  [░░░░░░░░░░░░░░░░]  0/12 features
INIT-007  Phase 7: Industry Extensions      planned    0%  [░░░░░░░░░░░░░░░░]  0/10 features

Overall: 5% (3/60 features completed)
```

### 6.7 `plan import-master-plan`

One-time import of the 60 features from `NEXT_GEN_QMS_MASTER_PLAN.md` into the database. Parses the feature tables and creates Initiative + Feature records with `legacy_id` mapped (E3-001 → FEAT-001, etc.).

---

## 7. API Endpoints

Internal dashboard CRUD, same pattern as existing RoadmapItem/PlanDocument endpoints.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/internal/plan/initiatives/` | GET | List initiatives with progress |
| `/api/internal/plan/initiatives/save/` | POST | Create/update initiative |
| `/api/internal/plan/features/` | GET | List features with filters |
| `/api/internal/plan/features/save/` | POST | Create/update feature |
| `/api/internal/plan/features/<id>/deps/` | POST | Add/remove dependencies |
| `/api/internal/plan/tasks/` | GET | List tasks with filters |
| `/api/internal/plan/tasks/save/` | POST | Create/update task |
| `/api/internal/plan/tree/<init_id>/` | GET | Full hierarchy tree |
| `/api/internal/plan/search/` | GET | Full-text search across all levels |
| `/api/internal/plan/blocked/` | GET | All blocked features |

---

## 8. Dashboard Integration

New "Planning" tab on the internal dashboard alongside existing Roadmap, Changes, Plans tabs.

### 8.1 Board View

Kanban-style board per initiative:

```
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│   Backlog   │   Planned   │ In Progress │   Blocked   │  Completed  │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ FEAT-053    │ FEAT-043    │ FEAT-042    │             │             │
│ Recurrence  │ Email notif │ Notification│             │             │
│ detection   │ ⚠ dep:042   │ system      │             │             │
│ P:medium    │ P:critical  │ P:critical  │             │             │
│             │             │ ████░░ 60%  │             │             │
│             │ FEAT-044    │             │             │             │
│             │ E-signature │ FEAT-045    │             │             │
│             │ P:critical  │ CAPA model  │             │             │
│             │             │ P:critical  │             │             │
│             │             │ ██░░░░ 33%  │             │             │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

### 8.2 Detail Panel

Click any feature to see the `plan show` output: full context, tasks, dependencies, change requests, progress.

### 8.3 Dependency Graph

Visual DAG rendering (upstream/downstream) for any feature. Uses the existing whiteboard SVG rendering engine.

---

## 9. Cross-Session Context Protocol

The primary use case: Eric starts a new Claude session, hands it a short ID, Claude immediately has full context.

### 9.1 Session Start

```
User: "Let's work on FEAT-042"

Claude runs: python manage.py plan show FEAT-042

Claude now knows:
- What this feature is
- What initiative it belongs to
- What's been completed (3/5 tasks)
- What's still open (TASK-120, TASK-121)
- What depends on this feature (FEAT-043, FEAT-055, FEAT-061)
- What standards apply
- What ChangeRequests exist
- The acceptance criteria
```

### 9.2 Session End

```
Claude completed TASK-120. Before ending:

1. Mark task completed:
   python manage.py plan update TASK-120 --status completed

2. Create next task's CR if starting tomorrow:
   python manage.py plan update TASK-121 --status in_progress

3. Update feature progress (auto-computed):
   Feature FEAT-042 is now 80% (4/5 tasks)

4. Check if any blocked features are now unblocked:
   python manage.py plan blocked
```

### 9.3 Grepability

Short IDs appear in:
- Planning database (queryable via `plan` command)
- `log.md` entries (referenced in change descriptions)
- ChangeRequest descriptions (link back to feature)
- Standards documents (`<!-- plan: FEAT-042 -->` hooks)
- CLAUDE.md (active work references)

`grep -r "FEAT-042"` across the codebase reconstructs the full web of references.

---

## 10. Import Strategy — Master Plan → Planning DB

The `NEXT_GEN_QMS_MASTER_PLAN.md` already defines ~60 features across 5 phases. The `import-master-plan` command maps them:

| Master Plan ID | → | Short ID | Initiative |
|---------------|---|----------|------------|
| E3-001 | → | FEAT-001 | INIT-003 |
| E3-002 | → | FEAT-002 | INIT-003 |
| ... | | | |
| E3-012 | → | FEAT-012 | INIT-003 |
| E4-001 | → | FEAT-013 | INIT-004 |
| ... | | | |
| E7-010 | → | FEAT-060 | INIT-007 |

The `legacy_id` field preserves the E3-001 mapping so both IDs work for grep. The `import-master-plan` command is idempotent — running it twice doesn't create duplicates (matches on `legacy_id`).

Initiatives map to phases:

| Phase | → | Short ID | Title |
|-------|---|----------|-------|
| Phase 3 | → | INIT-003 | Enterprise Foundation |
| Phase 4 | → | INIT-004 | Document Control & Supplier Quality |
| Phase 5 | → | INIT-005 | Training, Audit & Design Controls |
| Phase 6 | → | INIT-006 | Intelligence Layer |
| Phase 7 | → | INIT-007 | Industry-Specific Extensions |

Note: INIT-001 and INIT-002 are reserved for Phases 1 and 2 (already completed per the QMS roadmap).

---

## 11. Relationship to Existing Models

### 11.1 RoadmapItem (unchanged)

RoadmapItem remains the **public-facing** model. Features that should appear on the customer roadmap get a `roadmap_item_id` link. Not every feature is public — internal infrastructure, standards, and debt items stay internal.

```
Feature (FEAT-042, internal)  ──roadmap_item_id──>  RoadmapItem (public /roadmap/)
```

### 11.2 PlanDocument (unchanged)

PlanDocument remains for markdown specs, RFCs, and retros. Features can reference PlanDocuments in their description. No structural link needed — they serve different purposes.

### 11.3 ChangeRequest (unchanged)

ChangeRequest remains the execution-level record. Tasks link to ChangeRequests via `change_request_id`. The chain is:

```
Initiative → Feature → Task → ChangeRequest → ChangeLog → git commit
```

### 11.4 Compliance Check — `check_planning`

New compliance check registered in ALL_CHECKS:

```python
def check_planning():
    """Verify planning hygiene per PLN-001."""
    issues = []

    # Features in_progress for >30 days without task activity
    stale = Feature.objects.filter(
        status="in_progress",
        updated_at__lt=now() - timedelta(days=30)
    )
    if stale:
        issues.append(f"{stale.count()} features stale >30 days: {...}")

    # Features with circular dependencies
    # (validated on save, but check for data integrity)

    # Active initiatives with 0 features
    empty = Initiative.objects.filter(status="active", features__isnull=True)
    if empty:
        issues.append(f"{empty.count()} active initiatives with no features")

    # Completed features without completed_at timestamp
    no_ts = Feature.objects.filter(status="completed", completed_at__isnull=True)
    if no_ts:
        issues.append(f"{no_ts.count()} completed features missing timestamp")

    return {
        "status": "fail" if any_fail else "pass",
        "details": {...},
        "soc2_controls": ["CC9.1"]
    }
```

---

## 12. Standard — PLN-001

This system warrants its own standard (PLN-001: Planning & Feature Management), separate from RDM-001 which covers the public roadmap. PLN-001 would define:

- Model structure (Initiative, Feature, Task)
- Short ID convention and uniqueness rules
- Status lifecycle and valid transitions
- Dependency validation (no circular deps)
- Required fields per level
- Compliance check (`check_planning`)
- Cross-reference requirements (every Feature must link to at least one standard or ISO clause)

---

## 13. Implementation Plan

### Phase A: Models + Migration (1 session)

1. Add Initiative, Feature, Task models to `api/models.py`
2. Migration
3. Admin registration (read-only, for debugging)

### Phase B: Management Command (1 session)

1. `plan list`, `plan show`, `plan tree`, `plan search`
2. `plan deps`, `plan blocked`, `plan progress`
3. `plan import-master-plan`

### Phase C: API Endpoints (1 session)

1. CRUD for all three models
2. Dependency management endpoints
3. Tree/hierarchy endpoint
4. Search endpoint

### Phase D: Dashboard UI (1 session)

1. Planning tab with board view
2. Detail panel
3. Initiative progress overview
4. Dependency graph visualization

### Phase E: Standard + Compliance (1 session)

1. Write PLN-001 standard
2. Implement `check_planning` compliance check
3. Wire test hooks

---

## 14. Migration from Markdown Plans

After the system is built:

1. Run `plan import-master-plan` to seed from `NEXT_GEN_QMS_MASTER_PLAN.md`
2. Verify all 60 features imported with correct initiative mapping
3. Add dependencies between features (manual — requires domain knowledge)
4. Link existing RoadmapItems to features via `roadmap_item_id`
5. For active work, create tasks within features
6. `docs/planning/*.md` files remain as reference — the DB is now the source of truth for status tracking

---

## 15. Why This Design

**vs. Jira/Linear:** Those are SaaS products we'd depend on. This is in our database, queryable by our compliance system, integrated with our ChangeRequest chain, accessible via management command. No external dependency.

**vs. GitHub Issues:** No bidirectional link to our standards, no ISO clause tracking, no dependency DAG, no compliance check integration, no short IDs that survive across sessions.

**vs. Current markdown:** Markdown files can't answer "what features are blocked?" or "what's the progress on Phase 3?" or "show me everything related to FEAT-042." The planning system can.

**vs. Extending RoadmapItem:** RoadmapItem is public-facing and flat by design (RDM-001). Adding hierarchy and internal tracking to a public model violates separation of concerns.

The planning system fills the gap between "we have a strategic vision in markdown" and "we have per-change execution tracking in ChangeRequest." It's the missing middle layer.

---

*This is a living document. The models, command output, and API endpoints shown here are the design — implementation follows after approval.*
