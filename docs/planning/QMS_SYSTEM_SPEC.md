# Svend QMS System Specification

**Version:** 0.1 — Design Phase
**Date:** 2026-03-26
**Authors:** Eric + Claude
**Status:** SUPERSEDED by LOOP-001 (docs/standards/LOOP-001.md)
**Supersedes:** NEXT_GEN_QMS_MASTER_PLAN.md (strategic intent preserved, architecture replaced)

> **This document is superseded.** The authoritative specification for the closed-loop operating model is **LOOP-001** (`docs/standards/LOOP-001.md`). This file is retained for historical context. LOOP-001 formalizes the three modes, three mechanisms (Signals, Mode Transitions, Commitments), QMS Policy service, FMIS/Bayesian FMEA, and Dynamic Process Model that this spec introduced in narrative form.
**Lineage:** Juran Trilogy (Plan/Control/Improve) → Protzman → Shingo/Ohno → Deming PDSA → TWI

---

## 1. Thesis

Quality management software is stuck in 1995. Every QMS on the market — ETQ, MasterControl, Arena, Veeva — is a database of forms with approval workflows. The CAPA is a form. The audit is a form. The document is a file in a folder. The "closed loop" is a human remembering to update one form after closing another.

Svend replaces forms with a system that reasons. The QMS is a single closed loop:

```
Signal → Investigate → Standardize → Verify → (Signal) → ...
```

Every feature maps to one of three modes. If it doesn't fit Investigate, Standardize, or Verify, it doesn't belong.

---

## 2. The Three Modes

### 2.1 Investigate

A problem surfaces. You don't fill out a form — you open an investigation workspace and build understanding.

**Experience:** Notebook-like environment. Part narrative, part data, part analysis. The investigator constructs an argument: here is the evidence, here is the hypothesis, here is the test, here is the conclusion. The CAPA is not a text field labeled "corrective action" — it is a hypothesis that gets tested.

**Tools available within the workspace:**
- Root cause analysis (5-why, fishbone, fault tree)
- Evidence collection (data, photos, measurements, operator interviews)
- Bayesian hypothesis engine (prior probability, likelihood ratios, posterior)
- DOE design (factorial, CCD, Latin square, power analysis)
- Before/after analysis (200+ statistical tests)
- Counterfactual projection (pre-intervention trend extrapolation)
- AI critique (challenge the hypothesis, suggest alternative explanations)
- Historical pattern matching (has this happened before? what worked?)

**Outputs:**
- Tested hypothesis with statistical evidence
- Cause-effect relationship (feeds knowledge system when activated)
- Action items that transition to other modes

**What replaces:**
- NCR "root cause" text field
- CAPA form
- Effectiveness verification checkbox

### 2.2 Standardize

The fix becomes the new standard. The document builder creates or revises the standard work document, and training delivers it.

**2.2.1 Document Builder**

**Experience:** Structured authoring environment. Not a textarea. Not Word pasted into a browser.

**Core capabilities:**
- **Template system**: Document types (SOP, Work Instruction, Policy, Specification, Form) each have a default structure. AI suggests section headings based on document type and process context.
- **AI-assisted drafting**: Author provides bullet points, context, investigation reference. Anthropic drafts the content. Author edits, accepts, or regenerates.
- **Inline media**: Drag-and-drop photos with annotation tools (arrows, callouts, highlights). Critical for visual work instructions — the operator needs to see what "correct" looks like.
- **Process references**: Link steps to equipment, tooling, specifications, control parameters. These are not dead text — they connect to the data model.
- **Version control**: Every save is a revision. Diff view between versions. Approval workflow and e-signatures already exist (ElectronicSignature model, CFR Part 11 compliant).
- **Export**: Clean PDF and Word output. Professional formatting. Photos render correctly. This alone is a selling point — anyone who's built a work instruction with photos in Word knows the pain.
- **Investigation linkage**: Every document revision links to the investigation that motivated the change. Auditors can trace: "Why did this SOP change?" → investigation → evidence → root cause → trial results.

**LLM:** Anthropic. Quality matters for auditable documents. Cost per user won't approach $49/mo.

**2.2.2 Training Delivery**

When a document is published or revised, training is generated:
- Training requirement created, linked to the document and its version
- Affected roles assigned automatically (from document metadata)
- Training includes the document content + any supplementary material

**What replaces:**
- ControlledDocument textarea (rebuild the authoring experience)
- Manual training assignment
- Disconnected document-investigation relationship

### 2.3 Verify

Did the standard stick? Three verification types, each with a different purpose.

**2.3.1 Frontier Cards — Safety & 5S**

Observation-based safety auditing and 5S assessment. Already built (safety module). Zone metadata supports risk profiling, 5S baselining, scheduling intelligence.

- 19 safety observation items across 6 categories (S/AR/U rating with severity)
- 26 5S deficiency items across 5 pillars
- Operator interaction with comfort level
- Close-the-loop feedback
- Card-to-FMEA processing pipeline

**2.3.2 Process Confirmations (PCs) — Standard Work Verification**

A Process Confirmation is a structured gemba observation that answers two questions:

1. **Was the standard followed?**
2. **Did following the standard produce the expected outcome?**

The observer goes to the point of work, watches the operator perform the process, and records observations against the standard work document.

**Rules of engagement (adapted from STOP methodology):**
- ALWAYS interact with the operator — this is not surveillance
- Observe objectively — record what you see, not what you expect
- Don't lead or question the operator during the task
- Always explain what you're doing and why before starting
- Always look for what's going RIGHT first — acknowledge good practice before addressing gaps
- Never ask "why didn't you follow the process?" — that's interrogation, not observation

**Diagnostic matrix:**

| Standard followed? | Outcome correct? | Diagnosis | System response |
|---|---|---|---|
| Yes | Yes | System works | Record as confirmation. Score contributes to operator/process confidence. |
| No | — | Standard unclear, impractical, or training gap | → `revise_document` or `train` action. NOT the operator's fault. |
| Yes | No | Process design is broken — standard encodes a failed process | → Investigation opens. This is a high-value signal. |

**PC structure:**
- Linked to a specific ControlledDocument (the standard being confirmed)
- Linked to an operator (Employee)
- Linked to a process/area/zone
- Observation items derived from the document's key steps
- Each step: Followed (yes/no/NA) + Outcome (pass/fail/NA) + Notes
- Overall result + observer notes + improvement observations
- Photos (inline, linked to specific steps)

**Training threshold linkage:** If an operator's PC pass rate on a specific standard drops below a configurable threshold → system auto-generates a retraining requirement on that document. The retraining includes reflection (hansei) — operator feedback on what's unclear or impractical. Reflections feed back to the document author.

**2.3.3 Forced Failure Tests — Detection Verification**

A Forced Failure test deliberately creates conditions where the process should fail, then observes whether the detection controls catch it.

**Purpose:** Validate that the control plan actually works. FMEA says "detection control X catches failure mode Y with detection score 3." Does it really?

**Two modes:**

1. **Hypothesis-driven**: You have a specific failure mode from FMEA. You create the conditions (within safety constraints) and observe whether the detection system catches it. This validates or invalidates the FMEA detection score.

2. **Exploratory**: You exploit known weaknesses or boundary conditions to observe how the process fails. You're not testing a specific detection control — you're probing for unknown failure modes.

**Structure:**
- Linked to FMEARow (the failure mode being tested) OR free-form for exploratory
- Test plan: what conditions will be created, what's the expected detection response
- Safety review: confirmation that the test can be conducted safely
- Result: detected / not detected / partially detected
- Evidence: photos, data, measurements
- FMEA update: if detection failed, revise detection score and add/modify controls

**Relationship to PCs:**
- PCs verify the operator follows the standard (the human element)
- Forced Failures verify the system catches problems (the process element)
- Together they answer: "Is the standard being followed, AND does the system work when it is?"

---

## 3. Mode Transitions (Action Items)

CAPA action items are not tasks on a to-do list. They are transitions between modes. Each action type creates a specific artifact and moves the workflow.

| Action Type | Creates | From → To |
|---|---|---|
| `investigate` | Opens investigation workspace | Signal → Investigate |
| `design_experiment` | Opens DOE workspace within investigation | Stays in Investigate |
| `revise_document` | Opens doc builder with existing SOP + investigation link | Investigate → Standardize |
| `create_document` | Opens doc builder for new standard work | Investigate → Standardize |
| `train` | Creates training requirement linked to document + assigns roles | Standardize → Verify |
| `add_control` | Creates/updates FMEA row or control plan entry | Investigate → Standardize |
| `process_confirmation` | Schedules PC for the process at specified interval | Standardize → Verify |
| `forced_failure` | Schedules forced failure test for specific FMEA row | Standardize → Verify |
| `audit_zone` | Schedules frontier card audit (safety/5S) | Standardize → Verify |
| `monitor` | Sets up SPC monitoring or defines verification criteria + timeline | Standardize → Verify |

When a verification activity (PC, forced failure, frontier card) finds a gap, it generates a signal that enters Investigate. The loop closes.

---

## 4. Training as Loop Participant

Training is not a compliance checkbox. It is a feedback mechanism that validates standard work.

**The cycle:**

```
Document published
  → Training requirement created, roles assigned
  → Operator completes training
  → Operator provides reflection (hansei):
      - What was clear?
      - What was confusing?
      - What would you change about the instruction?
      - What's different from how you actually do it?
  → Reflections aggregated for document author
  → If pattern emerges (e.g., 4/6 say step 3 is unclear) → signal to revise document
  → Document revised → re-train only the changed section
```

**Competency model (TWI Job Instruction):**
- Level 0: None
- Level 1: Awareness (has seen the standard)
- Level 2: Supervised (can perform under guidance)
- Level 3: Competent (can perform independently)
- Level 4: Trainer (can teach others)

**PC-Training linkage:**
- PC results are per-operator, per-standard
- Configurable threshold (e.g., 80% pass rate over trailing 5 PCs)
- Below threshold → auto-generate retraining requirement
- Retraining includes reflection
- Not punitive — diagnostic. The standard might be the problem.

---

## 5. CI Health Score

Not "are your forms filled out" but "is the loop turning?"

### 5.1 What It Measures

| Indicator | Question | Data Source |
|---|---|---|
| Signal detection rate | Are problems being found before customers find them? | Frontier cards, PCs, forced failures, SPC, complaints |
| Investigation velocity | Are investigations producing tested fixes? | Investigation close rate, avg time, DOE usage |
| Hypothesis testing rate | Are fixes being tested, not just assumed? | % of investigations with statistical evidence |
| Standardization lag | When a fix is found, how fast does it become the standard? | Investigation close → document revision time |
| Training coverage | Are people trained on current standards? | % current, % with reflections, competency levels |
| Verification activity | Are PCs and forced failures happening at defined frequency? | Schedule adherence for PCs, frontier cards, forced failures |
| Recurrence rate | Are fixed problems staying fixed? | Repeat failure modes, repeat root causes across investigations |
| Standard work quality | Are standards being followed? Are they producing good outcomes? | PC pass rates (followed vs. outcome) |
| Detection capability | Do controls actually catch failures? | Forced failure test pass/fail rates vs. FMEA detection scores |

### 5.2 Scoring Philosophy

- Absence = failure. No management review in 12 months is a gap, not "no data."
- Staleness decays. Evidence has a half-life. Old data contributes less.
- Overdue items are worst. An overdue action means you knew and didn't act.
- The score can go down when you add data. That's the system working.
- Compliance evidence is an OUTPUT. The score measures improvement system health, not paperwork completion.

### 5.3 Audit Alignment

The health score dimensions map to ISO 9001 clauses, but the score is not organized by clause. An auditor can view the same data organized by clause for their purposes — that's the Auditor Portal view (read-only, time-limited token access). The organization sees improvement health. The auditor sees compliance evidence. Same data, different lens.

---

## 6. Auditor Portal

Read-only view of the same system, organized for external auditors.

**Access:** Time-limited token (ActionToken pattern). Quality manager generates a link, shares with auditor. No account required. Expires after the audit.

**View:**
- Clause-organized evidence (ISO 9001, IATF 16949, AS9100 — selectable)
- For each clause: relevant records, documents, training records, audit findings, investigation results
- Traceability chains: click an NCR → see the investigation → see the fix → see the document revision → see the training records → see the PC results confirming the fix held
- Health score trend (shows the system is improving over time)
- No edit capability. Pure evidence presentation.

**Why this matters:** The auditor doesn't need to learn Svend. They need to see evidence organized by the standard they're auditing. This is 90% of audit prep time eliminated.

---

## 7. Signal Sources

Signals enter the system from multiple sources. Each signal can trigger an investigation.

| Source | Type | Auto-trigger? |
|---|---|---|
| SPC out-of-control | Process | Not yet (quick path exists, defer forced automation) |
| Frontier card (AR/U finding) | Safety/5S | Manual — reviewer decides |
| Process confirmation gap | Standard work | Auto if pattern (threshold breach) |
| Forced failure detection miss | Process | Auto — detection gap is always investigated |
| Customer complaint | External | Manual — not all complaints warrant investigation |
| Audit finding (NC major/minor) | Audit | Auto for NC major, manual for minor |
| Supplier quality issue | Supply chain | Manual |
| Management review action | Leadership | Manual |
| Training reflection pattern | Competency | Auto if threshold (e.g., >50% report confusion on same step) |
| Operator report | Frontline | Manual |

---

## 8. Data Model Changes Required

### 8.1 New Models

| Model | Purpose | Mode |
|---|---|---|
| `Investigation` | Investigation workspace — replaces CAPA-as-form | Investigate |
| `InvestigationEntry` | Notebook entries within investigation (narrative, data, analysis, evidence, conclusion) | Investigate |
| `ActionItem` | Mode transition actions with type enum | Cross-mode |
| `ProcessConfirmation` | PC observation record | Verify |
| `PCObservationItem` | Individual step observation within a PC | Verify |
| `ForcedFailureTest` | Forced failure test record | Verify |
| `TrainingReflection` | Hansei response from operator | Standardize ↔ Verify |
| `ReadinessSnapshot` | Periodic CI health score snapshot | Reporting |
| `AuditorPortalToken` | Time-limited access token for external auditors | Reporting |

### 8.2 Extended Models

| Model | Changes | Why |
|---|---|---|
| `ControlledDocument` | Rich content model (sections, inline media, process references) | Document builder |
| `CAPAReport` | Link to Investigation, deprecate standalone workflow | Investigation replaces CAPA form |
| `TrainingRecord` | Add reflection fields, PC linkage | Training as feedback loop |
| `FMEARow` | Link to ForcedFailureTest results, auto-update detection scores | Forced failure validation |

### 8.3 Models NOT Changed

| Model | Why |
|---|---|
| `FrontierCard`, `FrontierZone` | Already built, already in the Verify mode |
| `ElectronicSignature` | Already CFR Part 11 compliant, no changes needed |
| `ComplianceCheck` | Internal infrastructure, separate from QMS user features |

---

## 9. Implementation Sequence

Not phased by ISO clause. Phased by what completes a usable loop.

### Layer 1: Investigation Workspace
- Investigation + InvestigationEntry models
- Notebook-like UI (extend A3/notebook pattern)
- Action items with type enum
- Bridge from existing NCR/CAPA to investigation
- Counterfactual analysis capability

### Layer 2: Document Builder
- Rich content model for ControlledDocument
- Structured authoring UI with AI assistance (Anthropic)
- Inline photo support with annotation
- PDF/Word export
- Investigation → document revision linkage

### Layer 3: Process Confirmations
- PC model linked to ControlledDocument
- Observation UI (mobile-friendly)
- Scoring and threshold system
- Training auto-trigger on threshold breach

### Layer 4: Training Reflection + Forced Failure
- TrainingReflection model
- Reflection UI within training completion flow
- Reflection aggregation and signal generation
- ForcedFailureTest model linked to FMEARow
- Test planning and result capture UI

### Layer 5: CI Health Score + Auditor Portal
- ReadinessSnapshot model and computation engine
- Dashboard widget (score + radar + trend + biggest gaps)
- Auditor portal with time-limited token access
- Clause-organized evidence view

---

## 10. Simulation & Interactivity — The Process Model

Every investigation doesn't just produce a conclusion — it refines a **lightweight causal process model**. Not a digital twin. A regression/Bayesian model calibrated from investigation data, DOE results, and verification outcomes. The model starts empty and gets smarter with each investigation.

### 10.1 How the Model is Built

The model is not designed upfront. It grows as a side effect of doing quality work:

| Activity | What It Contributes |
|---|---|
| Investigation with DOE | Input-output relationships with measured effect sizes |
| Investigation without DOE | Observational data, weaker but still calibration signal |
| Process Confirmation | Confirms or challenges model's prediction of process behavior |
| Forced Failure Test | Empirical detection probability for specific failure modes |
| SPC data | Ongoing process variability, drift patterns, shift magnitudes |
| FMEA review | Domain knowledge about what can go wrong (priors for Bayesian model) |

After 20 investigations on the same process, the model is rich enough to simulate. Early on, it acknowledges uncertainty — "I don't have enough data to predict this confidently."

### 10.2 Simulation in Investigate Mode — "What if?"

The investigation workspace gets a simulation panel with interactive widgets:

| Interaction | What Happens |
|---|---|
| **Parameter sweep** | Slider for input variable. Model predicts output distribution via Monte Carlo. "What if we change temperature from 180 to 190?" |
| **Failure injection** | "What if this input goes out of spec?" → model shows the cascade. Which outputs fail? By how much? |
| **Fix simulation** | "If we add this control, what's the predicted occurrence reduction?" → compare to current FMEA estimate |
| **Sensitivity analysis** | "Which inputs matter most for this output?" → rank by influence. Focus investigation on what moves the needle. |
| **Counterfactual** | After implementing a fix: run the model WITHOUT the fix, compare to actual post-fix results. Real effect size, not assumed. |

The investigator doesn't need to be a statistician. Sliders for inputs, distribution plots for outputs, confidence intervals for predictions. They explore, the system computes.

### 10.3 Simulation in Standardize Mode — "Will this standard work?"

Before publishing a document, the system can check the standard against the process model:

| Interaction | What Happens |
|---|---|
| **Tolerance analysis** | SOP says 180±5°C. Model says process is sensitive above 183°C. System flags the conflict before publication. |
| **Control plan validation** | With these inspection frequencies and sample sizes, what's the probability of a defect escaping? → operating characteristic curve generated from model. |
| **Training simulation** | Simulated process run where the trainee makes decisions. Model shows the consequence of each choice. Not a quiz — a simulated gemba. |

### 10.4 Simulation in Verify Mode — "Is it still working?"

Verification activities both consume and feed the model:

| Interaction | What Happens |
|---|---|
| **PC trend projection** | Pass rate is declining. "If trend continues, when do we breach the retraining threshold?" → project forward, recommend preemptive action. |
| **Forced failure planning** | FMEA says detection score is 3. Model predicts detection control catches this failure 85% of the time. Design the test to validate that specific prediction. |
| **SPC what-if** | "If the process mean shifts by 0.5σ, how many samples before we detect it?" → ARL analysis. "Is our control chart design adequate for the risk level?" |
| **Risk-based audit scheduling** | Model predicts which processes are drifting → audit those first, not on a calendar. Replaces static audit frequency with computed urgency. |

### 10.5 FMEA as Simulation (Not Committee Vote)

This is the single most differentiating capability. Every QMS platform and every FMEA facilitator on earth does this:

```
"OK team, what severity do we think? ... 7? 8? Let's say 7."
"Occurrence? ... We've seen it a few times... 4."
"Detection? ... We have an inspection... 3."
"RPN is 84. Next row."
```

That's opinion. Svend does this:

```
Severity: Model predicts this failure mode produces 2.3% scrap
          and a 15-minute line stop. Historical cost: $1,200/event.
          → Severity: 7 (computed from impact model)

Occurrence: 14 events in last 6 months across 12,000 units.
            Rate: 0.12%. Model predicts rate increases to 0.18%
            if supplier variability increases by 10%.
            → Occurrence: 3 (computed from historical data)

Detection: Forced failure test on 2026-02-15 showed detection
           control caught 6/8 injected failures (75%).
           → Detection: 5 (computed from empirical test)
           Previous team estimate was 3 — overconfident.

RPN: 105 (was 84 from consensus). Action required.
```

**Why nobody else can do this:** It requires the statistical engine, DOE capability, forced failure testing, and FMEA in the same system with the same data model. Svend has all four.

### 10.6 Implementation

Not a physics engine. The process model is:

1. **Regression models** calibrated from investigation data and DOE results (Svend already has these in DSW)
2. **Monte Carlo wrapper** that samples from input distributions and propagates through the model
3. **Bayesian updating** as new data arrives (Synara engine already exists)
4. **Interactive visualization** — sliders, distribution plots, sensitivity charts, confidence intervals

The flywheel: each investigation, each DOE, each PC, each forced failure adds calibration data. The system gets smarter the more you use it. This is the knowledge graph's activation path — not a static ontology, but a dynamic process model built from real failures and real fixes.

---

## 11. What This Is Not

- **Not a compliance system that claims to improve.** It is an improvement system that happens to satisfy auditors.
- **Not a form database.** There are no forms. There are investigations, documents, and observations.
- **Not optional infrastructure.** This is the core product experience for Enterprise QMS users.
- **Not a replacement for SPC, DSW, or existing statistical tools.** Those tools are consumed within investigations and simulations. The QMS is the workflow layer that gives statistical results operational meaning.
- **Not a digital twin.** The process model is lightweight, calibrated from real quality data, and acknowledges uncertainty. It doesn't pretend to be physics.

---

## 12. Open Questions

- [ ] Knowledge graph activation timing: the process model IS the graph's content — when does it become queryable beyond the individual process?
- [ ] PC scheduling: model-driven frequency (processes predicted to drift get more PCs) or fixed cadence?
- [ ] Document builder editor technology: ProseMirror/TipTap? Block-based? What supports inline photos + annotations best?
- [ ] Process model persistence: per-process? per-product line? How does it handle multiple related processes?
- [ ] Multi-standard support in health score: configurable weights per standard (ISO 9001 vs IATF vs AS9100)?
- [ ] SPC auto-trigger: deferred for now, quick path exists — define the quick path
- [ ] Mobile experience for PCs and frontier cards: responsive web or dedicated approach?
- [ ] Model confidence thresholds: when does the model have enough data to surface simulation results vs. showing "insufficient data"?
- [ ] FMEA transition: how to migrate from committee-voted scores to computed scores without invalidating existing FMEAs?
