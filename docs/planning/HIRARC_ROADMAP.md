# HIRARC ROADMAP — SVEND SAFETY

**Created:** 2026-03-28
**Status:** ACTIVE — co-designing with IRL standardization
**Standard:** SAF-001 (v1.0 approved 2026-03-21)
**Context:** Eric is standardizing a comprehensive safety observation program (DuPont STOP lineage, but comprehensive HIRARC). Svend Safety encodes it. Real documents will be ported in as the IRL system is designed.

---

## CURRENT STATE (v1.0 — shipped)

What SAF-001 v1.0 delivers today:

| Component | Status | Notes |
|-----------|--------|-------|
| FrontierZone (zone definition + risk profiling) | LIVE | 5 zone types, hierarchy, E-S-E-A-P controls |
| AuditSchedule + AuditAssignment | LIVE | Weekly publish, auditor rotation, 95% target |
| FrontierCard (20 safety + 25 5S items) | LIVE | S/AR/U ratings, C/H/M/L severity, operator interaction |
| Card-to-FMEA pipeline | LIVE | `process_card_to_fmea()` — AR/U findings become FMEARow entries |
| 5S Pareto aggregation | LIVE | 80/20 analysis, cross-feed to FMEA |
| Safety KPI dashboard | LIVE | Leading + lagging indicators, 30-day rolling |
| High-severity notification | LIVE | C/H findings → site admins via `notify()` |
| D-S-O-I-R audit cycle | LIVE | Define-Schedule-Observe-Intervene-Review |

**Not yet built (SAF-001 "Planned"):**
- Job Safety Analysis (JSA)
- Blue Tag (equipment-specific hazards)
- BBSO (behavior-based safety observations — the DuPont STOP core)
- Incident Report integration
- Scheduled safety reminders
- Auditor competency tracking (TRN-001 referenced, not wired)

---

## PHASE 1: JSA (Job Safety Analysis)

**What it is:** A task-based hazard identification vector. Breaks a job into sequential steps — for each step: what's the hazard, what's the control, what's the residual risk. Different from Frontier Cards (zone-based observation) — JSAs are proactive, written before the work starts.

**IRL alignment:** JSAs are the document workers sign before performing non-routine or high-risk tasks. Annual review cycle. Living documents that update when process changes.

### 1.1 Data Model (planned — awaiting IRL document review)

```
JSA
  ├── site (FK → Site)
  ├── title, job_description
  ├── department / area
  ├── zone (FK → FrontierZone, nullable — where the job is performed)
  ├── owner (FK → Employee — who maintains it)
  ├── review_cycle: annual | semi_annual | quarterly
  ├── next_review_date
  ├── status: draft | active | under_review | superseded | archived
  ├── previous_version (FK → JSA, nullable — revision chain)
  ├── ppe_required (JSON list)
  ├── training_required (JSON list — maps to TRN-001 competencies)
  ├── created_by, approved_by, approved_at
  └── is_active

JSAStep (ordered)
  ├── jsa (FK → JSA)
  ├── step_number
  ├── description (what the worker does)
  ├── hazards (JSON list — what can go wrong)
  ├── controls (JSON — keyed by E-S-E-A-P hierarchy level)
  ├── residual_risk: low | medium | high
  ├── severity: C | H | M | L (maps to FMEA via SEVERITY_TO_FMEA)
  └── notes
```

### 1.2 Integration Points

| Touchpoint | How |
|---|---|
| **FMEA** | Steps with residual_risk > threshold → FMEARow (same pipeline pattern as `process_card_to_fmea`) |
| **FrontierZone** | JSA linked to zone — auditors see applicable JSAs when observing zone |
| **Frontier Card** | Operator interaction: "Is there a current JSA for this task?" — close-the-loop |
| **Training (TRN-001)** | JSA review = training record. Worker sign-off = competency evidence |
| **Action Items** | JSA review findings → ActionItem for stale controls, missing specs |

### 1.3 API Endpoints (planned)

```
GET,POST   /api/safety/jsas/                    — list/create
GET,PUT    /api/safety/jsas/<id>/                — detail/update
POST       /api/safety/jsas/<id>/review/         — mark reviewed (creates new version if revised)
POST       /api/safety/jsas/<id>/process/        — push high-risk steps to FMEA
GET        /api/safety/jsas/overdue/             — JSAs past next_review_date
```

### 1.4 Open Questions

- [ ] What does the IRL JSA form look like? (Eric porting document)
- [ ] Risk scoring per step: simple residual_risk (L/M/H) or mini-FMEA (S×L per step)?
- [ ] Sign-off model: digital signatures per worker per JSA, or training-record-based?
- [ ] Revision policy: new version on any change, or only on substantive changes?

---

## PHASE 2: ANNUAL REVIEW REMINDERS

**What it is:** Scheduled sweep tasks that fire notifications when safety documents approach or exceed their review dates. Uses existing `syn.sched` + `notify()` infrastructure.

### 2.1 Reminder Matrix

| Reminder | Sweep Frequency | Advance Warning | Recipient | NotificationType |
|---|---|---|---|---|
| JSA review due | Monthly (1st) | 30 days before `next_review_date` | JSA owner | `REVIEW_DUE` |
| JSA overdue | Weekly (Mon) | Past `next_review_date` | JSA owner + site safety lead | `ACTION_DUE` |
| Zone audit overdue | Weekly (Mon) | Per `FrontierZone.is_overdue` | preferred_auditors | `ACTION_DUE` |
| Unprocessed cards (>48h) | Daily | 48h since `audit_date` | Card auditor | `ACTION_DUE` |
| Weekly schedule not published | Weekly (Mon noon) | If no AuditSchedule for current week | Area manager | `AUDIT_SCHEDULED` |

### 2.2 Implementation Pattern

Same as Harada daily reminders — a handler registered in `svend_tasks.py`, a schedule in `SVEND_SCHEDULES`, and the handler calls `notify()` for each match.

```
safety.jsa_review_reminders    → cron: 0 8 1 * *   (monthly, 1st at 08:00 UTC)
safety.overdue_sweep           → cron: 0 8 * * 1   (weekly, Monday 08:00 UTC)
safety.unprocessed_card_nudge  → cron: 0 10 * * *  (daily 10:00 UTC)
```

### 2.3 Escalation

- 30-day warning → owner only
- Overdue → owner + site safety lead
- 60 days overdue → owner + safety lead + site manager (if role exists)

---

## PHASE 3: BBSO (Behavior-Based Safety Observations)

**What it is:** The DuPont STOP core — structured observation of worker behavior (not conditions). Frontier Cards already capture some of this (body_position, ppe, operator interaction), but BBSO is a dedicated vector focused on at-risk behaviors with positive reinforcement.

**IRL alignment:** This is the heart of the comprehensive program. DuPont STOP tracks safe/at-risk behavior ratios over time. The observation is a conversation, not an inspection.

### 3.1 How It Differs from Frontier Cards

| Dimension | Frontier Card | BBSO |
|---|---|---|
| Focus | Conditions + behaviors (mixed) | Behaviors only |
| Trigger | Scheduled zone audit | Any time, any place |
| Duration | ~20 min structured walkthrough | ~5 min focused observation |
| Outcome | AR/U → FMEA rows | At-risk behavior → coaching conversation → trend data |
| Positive tracking | `positive_observations` field | Core feature — safe behavior ratio is the KPI |
| Volume | 2+ per supervisor per week (scheduled) | High frequency, low overhead (culture metric) |

### 3.2 Data Model Sketch

```
BBSOObservation
  ├── observer (FK → Employee)
  ├── site, zone (nullable — can happen anywhere)
  ├── observation_date
  ├── task_observed (what the worker was doing)
  ├── behaviors (JSON list):
  │     [{category, behavior, rating: safe|at_risk, notes}]
  ├── conversation_held: boolean
  ├── conversation_notes
  ├── positive_reinforcement (what safe behaviors were acknowledged)
  └── followup_needed: boolean
```

### 3.3 KPIs (leading indicators)

- **Safe behavior ratio**: safe / (safe + at_risk) per period — THE metric
- **Observation frequency**: observations per supervisor per week
- **Conversation rate**: % of observations with conversation_held = True
- **Category trends**: which behavior categories are improving/declining

---

## PHASE 4: INCIDENT REPORT INTEGRATION

Post-incident → root cause → FMEA recalibration. Ties to INC-001. Deferred until Phases 1-3 are stable.

---

## PHASE 5: BLUE TAG (EQUIPMENT HAZARDS)

Equipment-specific hazard tagging. Lower priority — most equipment hazards surface through Frontier Card audits and JSAs. Separate tracking needed for lockout/tagout compliance and equipment-specific risk registers.

---

## IRL DOCUMENT INTAKE LOG

As Eric ports real-world documents, they'll be logged here with what they informed:

| Date | Document | What It Informed |
|------|----------|------------------|
| — | *(awaiting first document)* | — |

---

## DESIGN PRINCIPLES

1. **Every HI vector feeds the same FMEA.** Frontier Cards, JSAs, BBSOs, incident reports — all roads lead to the central risk register. One risk picture.
2. **Immutability after submission.** Same as Frontier Cards — write-once, append-only. Revisions create new records with `previous_version` FK.
3. **Positive observation is not optional.** DuPont STOP's insight: if you only track what's wrong, you train people to hide. Safe behavior ratios, positive reinforcement, and conversation are first-class data.
4. **The observation is a conversation.** An audit that doesn't include talking to the operator missed the point. Frontier Cards already enforce this (3 operator questions). BBSO makes it the entire point.
5. **Annual review is a minimum.** JSAs get annual review by default, but high-risk jobs or process changes trigger immediate review. The system should make review easy enough that people do it.
6. **Reminders are escalating, not nagging.** 30-day warning, overdue nudge, escalation. Three levels, not a daily firehose.
