**LOOP-001: CLOSED-LOOP OPERATING MODEL**

**Version:** 0.1
**Status:** DESIGN
**Date:** 2026-03-27
**Authors:** Eric + Claude (Systems Architect)
**Supersedes:** QMS_SYSTEM_SPEC.md (strategic intent preserved, mechanisms formalized here)
**Lineage:** Deming PDSA → Juran Trilogy → TWI Job Instruction → David Mann Daily Management → Protzman Integrated Framework → Svend
**Compliance:**
- ISO 9001:2015 §10.2 (Nonconformity and Corrective Action)
- ISO 9001:2015 §10.3 (Continual Improvement)
- ISO 9001:2015 §6.1 (Actions to Address Risks and Opportunities)
- ISO 9001:2015 §7.2 (Competence)
- ISO 9001:2015 §7.5 (Documented Information)
- ISO 9001:2015 §9.1 (Monitoring, Measurement, Analysis, Evaluation)
- IATF 16949:2016 §10.2.3 (Problem Solving)
- IATF 16949:2016 §10.2.4 (Error-Proofing)
- 21 CFR Part 820 §820.90 (Nonconforming Product)
- AS9100D §8.7 (Control of Nonconforming Outputs)
**Related Standards:**
- CANON-001 ≥ 3.0 (System Architecture — three-layer model)
- CANON-002 ≥ 1.0 (Evidence Methodology — tool integration contracts)
- QMS-001 ≥ 1.6 (Quality Management System — tool specifications)
- TRN-001 ≥ 1.0 (Training Competency)
- RISK-001 ≥ 1.0 (Risk Registry)

---

## **1. SCOPE AND PURPOSE**

### **1.1 Purpose**

LOOP-001 defines the operating model for Svend's quality management system. It replaces the traditional form-based CAPA process with a closed-loop system of three modes — Investigate, Standardize, Verify — connected by formally defined mechanisms.

**Core Principle:**

> Quality management is not form completion. It is a closed loop: detect a signal, investigate the cause, standardize the fix, verify it holds. The loop IS the preventive action system. A system where the loop turns faster, with more evidence and less opinion, is a better quality system. Everything in this standard serves that loop.

### **1.2 Scope**

This standard covers:
- The three operating modes and their semantics (§2)
- Three formally defined mechanisms: Signals, Mode Transitions, Commitments (§3)
- QMS Policy service — org-defined rules that inform system behavior (§4)
- CAPA as generated compliance artifact, not workflow (§5)
- Training as loop participant with hansei reflection (§6)
- Process Confirmations and Forced Failure Tests (§7)
- The FMEA Reformation — Bayesian S/O/D replacing committee votes (§8)
- Dynamic Process Model — quantitative knowledge graph (§9)
- CI Readiness Score — "is the loop turning?" (§10)
- Auditor Portal — compliance evidence view (§11)
- Bias detection and accountability transparency (§12)

This standard does NOT cover:
- Individual tool behavior (see QMS-001, CANON-001)
- Evidence weighting methodology (see CANON-002)
- Statistical analysis engine internals (see DSW-001, STAT-001)

### **1.3 Design Constraints**

1. **The user orchestrates, the system executes.** The system does not compose simulations, infer root causes, or auto-trigger investigations. It exposes gaps, enforces accountability, and maintains the evidence chain. Critical thinking is not subordinated to AI or automation.

2. **Human-in-the-loop for all triggers (v1).** No signal automatically creates an investigation. Org-defined policies set thresholds, but a human decides to act. Automation is deferred pending longitudinal study of actual usage patterns (see §4.6).

3. **Evidence over opinion.** Where the system can compute a value from data (detection rate, occurrence rate, consequence severity), it does. Where it cannot, it says so explicitly — it does not fill gaps with defaults or AI-generated estimates.

4. **The model should be lossy.** The Dynamic Process Model (§9) does not pretend to be truth. It shows where knowledge exists, where it doesn't, and how confident the existing knowledge is. Its value is in exposing gaps, not in predicting outcomes.

5. **Policy is structured data, not documentation.** Org-defined policies are machine-readable models that inform system behavior. Human-readable documentation is auto-generated as a compliance artifact. No drift between what's enforced and what's documented.

### **1.4 Information-Theoretic Foundation**

The closed loop is an information engine. Every action — every investigation, every PC, every forced failure test, every DOE — is evaluated by how much uncertainty it reduces, not by whether a form was filled out. Shannon's information theory (1948) provides the mathematical frame. Shewhart's control chart (1931) is a special case. The unification is overdue.

**The core identity:** The value of an observation is the entropy it eliminates.

A forced failure test that confirms 8/8 detection tells you almost nothing — the posterior barely moves. A test that finds 6/8 detection compresses significant uncertainty about the control's effectiveness. The "failure" is the high-information event. The system must recognize this: surprising results are more valuable than confirming ones, and the system should surface which actions would yield the most information next.

**Applied to each mode:**

| Mode | Information-theoretic purpose | Metric |
|---|---|---|
| **Investigate** | Reduce entropy over the process model. Every tool use, DOE, and evidence observation is selected to maximize expected information gain over the causal graph. | Mutual information between candidate action and unknown process parameters |
| **Standardize** | Encode compressed knowledge as standard work. A good standard is a low-entropy representation of "how to run this process" — it minimizes the information the operator needs to reconstruct correct behavior. | Entropy of the standard-to-practice channel (how much ambiguity remains after reading the SOP) |
| **Verify** | Measure the channel capacity of the standard-to-reality transmission. PCs measure: is the standard being received (followed)? Is the channel working (outcome correct)? Forced failures measure: is the detection channel operative? | Mutual information between true process state and observed verification signal |

**Applied to the process model (§9):**

The Dynamic Process Model is an entropy map of the organization's knowledge about a process. Calibrated edges (from DOEs, investigations) are low-entropy regions — we know what X does to Y with measured confidence. Uncalibrated edges are high-entropy regions — we think X affects Y but don't know how much.

The model's primary diagnostic is: **where is entropy highest, and which action would reduce it most?**

```
Expected information gain of action A = H(model) - E[H(model | outcome of A)]
```

If running a DOE on temperature × feed rate would reduce model entropy by 2.1 bits, and running one on coolant flow would reduce it by 0.4 bits, the investigator should run the temperature DOE first. The system computes this. The user decides.

This is not optimization for its own sake. It is the mathematical expression of "don't waste investigation effort on things you already know." Every hour an engineer spends confirming the obvious is an hour not spent discovering the unknown.

**Applied to FMIS (§8):**

The Bayesian S/O/D posteriors ARE entropy measures:
- A Beta(1,1) prior on detection has maximum entropy — we know nothing. Every forced failure test reduces this entropy.
- A Dirichlet(1,1,1,1,1) prior on severity has maximum categorical entropy. Each consequence observation reduces it.
- The posterior credible interval width IS a measure of remaining uncertainty. Narrow interval = low entropy = we know this. Wide interval = high entropy = investigate here.

When the FMIS view (§16.5) shows a wide posterior bar with "??" for no data, it is showing the user where entropy is concentrated. The system is saying: "This is where your next action should be."

**Applied to verification (§7):**

A Process Confirmation where the standard is followed and the outcome is correct yields near-zero information — it confirms the prior with high probability. The entropy reduction is minimal.

A PC where the standard is followed but the outcome is wrong yields MAXIMUM information — the standard faithfully encodes a broken process. This is the highest-information observation possible in Verify mode, because it tells you something no other signal source can: the documented knowledge itself is wrong.

The diagnostic matrix (§7.1) already captures this intuitively:

| Observation | Entropy reduction | Why |
|---|---|---|
| Followed + correct outcome | Low | Confirms prior (standard works). Expected result. |
| Not followed + any outcome | Medium | Reveals training/clarity gap. Somewhat expected for new standards. |
| Followed + wrong outcome | **Maximum** | Reveals the standard itself is wrong. Highly surprising. Highly informative. |

**Applied to the APC Frontiers (§13):**

- **Frontier 1** (info-theoretic charts): directly applies — chart design that maximizes mutual information per sample instead of minimizing ARL.
- **Frontier 2** (reaction plan stability): loop gain analysis is information-theoretic — an overreactive operator ADDS entropy to the process by amplifying noise.
- **Frontier 3** (Cpk as distribution): Wasserstein drift between capability distributions is a measure of distributional entropy change.
- **Frontier 5** (Bayesian fault classifier): the classifier's posterior concentration on a fault mode IS the entropy collapsing around a diagnosis. Detection = entropy reduction. Diagnosis = identifying which entropy was reduced.

**The key finding from APC research (documented in ADVANCED_PROCESS_CONTROL.md):** The EWMA forgetting factor λ and the Bayesian classifier forgetting factor α are the same mathematical object — exponential forgetting at different levels of the inference hierarchy. The optimal forgetting rate ≈ 1/τ where τ is mean time between fault transitions. Frontiers 1 and 5 are the same optimization problem: maximize information transmission rate through a noisy, non-stationary channel.

**What this means for the system:**

1. Every surface that shows a posterior distribution (FMIS, process model, fault classifier) is showing an entropy map. The user learns to read wide bars as "investigate here" and narrow bars as "this is known."

2. The process model gap exposure (§9.4) is not a feature list — it is the system computing expected information gain for candidate investigations and surfacing the highest-gain opportunities.

3. The CI Readiness Score (§10) should include an information-theoretic dimension: "How much entropy has the loop reduced this quarter?" A system that runs lots of PCs but never discovers anything surprising has low information throughput — the verification channel capacity is being wasted on confirmatory observations.

4. The report engine (§5.2) should include the information narrative: "This investigation reduced process model entropy by X bits. The highest-information finding was Y (posterior shifted from Z to W)." This is the mathematical proof that the investigation was worth doing.

This framing is NOT optional philosophy. It is the engineering principle that connects every component of LOOP-001 into a coherent system. The loop exists to reduce entropy about how processes work. Everything else is mechanism.

---

## **2. THE THREE MODES**

### **2.1 Mode Definitions**

| Mode | Purpose | Entry condition | Exit condition | Primary artifacts |
|---|---|---|---|---|
| **Investigate** | Build understanding of a problem through structured reasoning | Signal received and triaged | Concluded hypothesis with evidence, action commitments created | Investigation workspace, causal graph, evidence chain, tested hypothesis |
| **Standardize** | Encode the fix as standard work and deploy it | Mode transition from Investigate (revise_document, create_document, add_control) | Document published, training assigned | Controlled documents, FMEA updates, training requirements |
| **Verify** | Confirm the standard holds and the loop is working | Mode transition from Standardize (train, process_confirmation, forced_failure, monitor) | Verification evidence collected, either confirming or generating new signal | PC results, forced failure results, SPC data, frontier card observations, training records with reflections |

### **2.2 The Loop as Prevention**

The "P" in traditional CAPA (Preventive Action) is the loop itself. Specifically:

- **Forced failure tests** (§7.2) proactively probe detection controls before customers find failures
- **Process confirmations** (§7.1) detect standard work drift before it produces defects
- **FMEA reviews** with computed S/O/D (§8) identify emerging risks from data, not from annual re-scoring meetings
- **SPC monitoring** detects process shifts in real time
- **Frontier card observations** catch safety and 5S degradation

When an auditor asks "where are your preventive actions?", the answer is the Verify mode artifact chain. This framing MUST be explicit in generated compliance reports (§5).

### **2.3 Mode Invariants**

1. Every investigation MUST conclude with either a tested hypothesis or an explicit statement of insufficient evidence. "Root cause unknown" is a valid conclusion if the evidence is documented.
2. Every document revision MUST link to the investigation that motivated it. Orphan revisions (no investigation link) are flagged by the readiness score (§10).
3. Every verification activity MUST link to the standard it verifies. A PC without a linked ControlledDocument is invalid.
4. Mode transitions are system mechanics, not user decisions. When a commitment of type `revise_document` is fulfilled, the system creates the transition. The user decides WHAT to do (via commitments). The system decides HOW to record it (via mode transitions).

---

## **3. THREE MECHANISMS**

The system operates through three formally defined mechanisms. They are distinct objects with distinct lifecycles and distinct purposes.

### **3.1 Signals**

A Signal is an event that demands attention. It is the entry point to the loop.

**Definition:** A Signal is a record that something has been observed which may require investigation. It carries:
- **Source**: what generated it (SPC violation, PC threshold breach, customer complaint, audit finding, forced failure miss, frontier card finding, operator report, management review action, training reflection pattern)
- **Severity classification**: per org-defined QMS Policy (§4)
- **Triage state**: untriaged → acknowledged → investigating → resolved → dismissed
- **Linked artifacts**: the source record(s) that generated the signal (SPC result UUID, complaint UUID, etc.)
- **Resolution**: what happened — investigation opened, existing investigation linked, dismissed with reason

**Behavioral rules:**
- Signals are NEVER auto-created in v1. A human or a policy-defined threshold creates them. The system surfaces the CONDITIONS for a signal (e.g., "SPC rule violation detected") but does not create the Signal record without human confirmation.
- A Signal MUST be resolved. Unresolved signals degrade the readiness score (§10). Dismissing a signal requires a reason.
- A Signal MAY link to an existing investigation (the problem is already being worked) rather than creating a new one.
- Signal sources are defined in QMS Policy (§4). The system does not have a hardcoded list of what constitutes a signal.

### **3.2 Mode Transitions**

A Mode Transition is a structural linkage between artifacts across modes. It is a system mechanic, not a human decision.

**Definition:** When a Commitment (§3.3) of a specific type is fulfilled, the system creates the corresponding artifact in the target mode and records the transition. The transition is immutable — it is part of the audit trail.

**Transition types and their mechanics:**

| Transition type | Trigger | System creates | From → To |
|---|---|---|---|
| `revise_document` | Commitment fulfilled | Opens document editor with existing ControlledDocument + investigation link | Investigate → Standardize |
| `create_document` | Commitment fulfilled | Opens document editor for new document + investigation link | Investigate → Standardize |
| `add_control` | Commitment fulfilled | Creates or links FMEA row with investigation-sourced data | Investigate → Standardize |
| `train` | Document published | Creates TrainingRequirement linked to document version + assigns roles per document metadata | Standardize → Verify |
| `process_confirmation` | Commitment fulfilled | Creates PC schedule for specified process/standard/interval | Standardize → Verify |
| `forced_failure` | Commitment fulfilled | Creates ForcedFailureTest plan linked to specific FMEA row | Standardize → Verify |
| `monitor` | Commitment fulfilled | Creates SPC monitoring configuration or verification criteria with timeline | Standardize → Verify |
| `audit_zone` | Commitment fulfilled | Schedules frontier card audit for specified zone | Standardize → Verify |

**Behavioral rules:**
- Mode transitions are recorded as immutable audit entries with: transition_type, source_artifact_id, target_artifact_id, triggered_by_commitment_id, timestamp.
- The `train` transition is the ONLY auto-triggered transition. When a ControlledDocument transitions to `approved` status, the system auto-creates training requirements per document metadata (roles, sites). All other transitions require a fulfilled commitment.
- Mode transitions do NOT have owners or due dates. They are structural. The Commitment has the owner and due date.

### **3.3 Commitments (Accountability Contracts)**

A Commitment is a bilateral contract between a person and the organization, per David Mann's accountability system.

**Definition:** A Commitment records:
- **Owner**: the person who commits to doing the work
- **What**: description of the specific deliverable (not a vague task — a concrete artifact or outcome)
- **Due date**: when it will be done
- **Preconditions**: what the owner needs from the organization to fulfill the commitment (resources, data, access, decisions from others)
- **Status**: open → in_progress → fulfilled → broken → cancelled
- **Transition type**: if this commitment, when fulfilled, triggers a mode transition (§3.2), which type
- **Source**: what created this commitment — investigation conclusion, management review, kaizen charter, manual
- **Source investigation**: FK to Investigation (nullable — not all commitments come from investigations)
- **Linked artifacts**: the target artifact(s) created when the commitment is fulfilled

**Behavioral rules:**

1. **Bilateral contract.** A commitment is not a task assigned to someone. It is a contract. The owner commits to the deliverable. The organization commits to the preconditions. If the preconditions are not met, the commitment is not "late" — it is blocked, and the blocker is visible.

2. **Visibility.** All commitments are visible on the accountability dashboard. Due today, overdue, blocked — with precondition status. This is reviewed at standup frequency. The system does NOT send nag emails — it makes reality visible.

3. **Broken commitments are system signals.** A commitment that is not fulfilled by the due date is not a performance issue. It is a signal. Why? Missing resources? Unclear scope? Wrong person? Prerequisite commitment not met? The PATTERN of broken commitments is diagnostic — it tells you where the organization's improvement system is failing.

4. **Fulfillment triggers mode transition.** When a commitment with a transition_type is marked `fulfilled`, the system executes the corresponding mode transition (§3.2) and creates the target artifact with proper linkage.

5. **Commitments chain.** A commitment from an investigation (Investigate mode) may create a document revision commitment (Investigate → Standardize). Completing that document creates a training commitment (Standardize → Verify). Completing training creates a PC scheduling commitment (Verify). The chain IS the loop, made visible.

6. **No ghost commitments.** Every commitment MUST have an owner and a due date. A commitment without a due date is not a commitment — it is a wish. The system rejects commitments without these fields.

---

## **4. QMS POLICY SERVICE**

### **4.1 Purpose**

QMS Policy is a global service that stores org-defined rules as structured data. These rules inform system behavior (thresholds, escalation paths, required fields, review frequencies). The system auto-generates human-readable policy documentation from this data.

### **4.2 Policy Model**

A QMS Policy record contains:
- **Scope**: what area this policy applies to (signal_thresholds, pc_configuration, training_requirements, review_frequencies, fmea_methodology, document_control, etc.)
- **Rule key**: machine-readable identifier (e.g., `pc.retraining_threshold`)
- **Parameters**: structured data (JSON) — the actual rule configuration
- **Linked standard**: which ISO/IATF/AS clause this policy satisfies (for audit traceability)
- **Effective date**: when this policy version takes effect
- **Approved by**: who authorized this policy
- **Version**: policy revision number
- **Tenant**: org-scoped (multi-tenant)

### **4.3 Policy Informs, Humans Decide (v1)**

In v1, policies define thresholds and surface conditions. They do NOT auto-trigger actions. Examples:

| Policy rule | What the system does | What the system does NOT do |
|---|---|---|
| `pc.retraining_threshold = 0.80 over trailing 5` | Surfaces "Operator X is below threshold on Standard Y" on accountability dashboard | Does NOT auto-create training requirement |
| `signal.spc_rule_violation = true` | Highlights OOC points in SPC view with "Create Signal?" prompt | Does NOT auto-create Signal record |
| `fmea.review_frequency_months = 6` | Surfaces "FMEA X is due for review" on dashboard | Does NOT auto-create review |

### **4.4 Policy as Controlled Document**

Every policy version is rendered as a human-readable document (using the document editor engine) and stored as a ControlledDocument. This means:
- Policy changes follow the same approval workflow as any controlled document
- Auditors can pull the policy that was in effect at any point in time
- Policy revision history is immutable

### **4.5 Policy Controls**

The policy service includes guards against control logic issues:

1. **Cooldown rules**: A retraining signal for the same operator+standard pair cannot fire again within `cooldown_days` of the previous one.
2. **Escalation rules**: If `escalate_after_n_operators` operators fail the same standard within `escalation_window_days`, the system surfaces a `revise_document` signal instead of individual retraining signals.
3. **Conflict detection**: If two policies contradict (e.g., one requires monthly review, another says quarterly), the system flags the conflict at policy save time.

### **4.6 Control Service Architecture**

The QMS Policy service evaluates org-defined rules against system events using a hybrid real-time/aggregate model.

**4.6.1 Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    ToolEventBus                               │
│  (spc.signal, pc.completed, fmea.row_updated, rca.created,  │
│   complaint.created, forced_failure.completed, ...)          │
└──────────────────────────┬──────────────────────────────────┘
                           │ subscribes
                           ▼
              ┌─────────────────────────┐
              │    PolicyEvaluator      │  ← Real-time evaluation
              │  (ToolEventBus handler) │
              └────────────┬────────────┘
                           │ reads
                           ▼
              ┌─────────────────────────┐
              │    QMSPolicy records    │  ← Org-defined rules (structured data)
              │  (scope, rule_key,      │
              │   parameters, tenant)   │
              └────────────┬────────────┘
                           │ produces
                           ▼
              ┌─────────────────────────┐
              │   PolicyCondition       │  ← Surfaced condition (NOT a Signal)
              │  (condition_type,       │
              │   severity, context,    │
              │   policy_rule_ref)      │
              └────────────┬────────────┘
                           │ displayed on
                           ▼
              ┌─────────────────────────┐
              │ Accountability Dashboard │
              │  Human reviews → may    │
              │  create Signal (§3.1)   │
              └─────────────────────────┘


              ┌─────────────────────────┐
              │  PolicySweepEvaluator   │  ← Aggregate evaluation (scheduled)
              │  (Tempora daily task)   │
              └────────────┬────────────┘
                           │ reads all policies with scope = aggregate
                           │ queries data (training coverage, review dates,
                           │   recurrence patterns, commitment fulfillment)
                           │ produces PolicyConditions
                           ▼
              (same PolicyCondition → Dashboard flow)
```

**4.6.2 PolicyEvaluator (Real-Time)**

Subscribes to ToolEventBus. When an event fires, evaluates all active policy rules whose `scope` matches the event type.

**Example evaluation flow:**
1. Event: `pc.completed` fires with payload `{operator_id, standard_id, pass_rate, items}`
2. PolicyEvaluator queries: all QMSPolicy records where `scope = "process_confirmation"` for this tenant
3. For rule `pc.retraining_threshold`: check trailing `window_size` PCs for this operator+standard
4. If pass rate < threshold AND no active cooldown: create PolicyCondition
5. For rule `pc.escalation`: check if `escalate_after_n_operators` have breached threshold on this standard within `escalation_window_days`
6. If escalation threshold met: create PolicyCondition with `condition_type = "standard_revision_needed"` instead of individual retraining conditions

**Real-time rule types:**

| Rule scope | Triggered by event | Evaluates |
|---|---|---|
| `process_confirmation` | `pc.completed` | Operator pass rate vs threshold, escalation to standard revision |
| `spc_monitoring` | `spc.signal` | OOC count, rule violation type, FMIS row linkage |
| `forced_failure` | `forced_failure.completed` | Detection miss → always surfaces condition |
| `complaint` | `complaint.created` | Severity classification, repeat product/process |
| `training` | `training.completed` | Reflection submitted (completeness check) |

**4.6.3 PolicySweepEvaluator (Aggregate)**

Runs as a Tempora scheduled task (daily by default, configurable). Evaluates conditions that require querying across records rather than responding to a single event.

**Aggregate rule types:**

| Rule scope | Evaluates | Surfaces |
|---|---|---|
| `review_frequency` | FMEA review dates, management review dates, document review dates | "FMEA X overdue for review by N days" |
| `training_coverage` | % of employees trained on current document versions | "Standard Y: 4 of 12 operators not trained on current version" |
| `recurrence` | Repeat failure modes across investigations within window | "Failure mode Z has appeared in 3 investigations in 6 months" |
| `commitment_fulfillment` | Overdue commitments, broken commitment patterns | "Team A: 5 overdue commitments, oldest 14 days" |
| `calibration` | MeasurementEquipment due/overdue dates | "Equipment X calibration overdue by 30 days" |
| `verification_schedule` | PC and forced failure schedule adherence | "Process Y: 0 of 4 scheduled PCs completed this month" |

**4.6.4 PolicyCondition Model**

A PolicyCondition is an intermediate object — the system noticed something. It is NOT a Signal. A human reviews conditions and decides which warrant investigation.

| Field | Type | Purpose |
|---|---|---|
| `id` | UUID PK | |
| `tenant` | FK → Tenant | Org scope |
| `condition_type` | CharField | Category: `retraining_needed`, `standard_revision_needed`, `review_overdue`, `recurrence_detected`, `detection_gap`, `threshold_breach`, `schedule_miss`, `calibration_overdue` |
| `severity` | CharField | `info` / `warning` / `critical` — derived from policy rule parameters |
| `title` | CharField | Human-readable summary |
| `context` | JSONField | Structured data: affected entities, measurements, thresholds, evidence |
| `policy_rule` | FK → QMSPolicy | Which policy rule generated this condition |
| `source_event` | CharField (nullable) | ToolEventBus event name that triggered evaluation (null for aggregate) |
| `status` | CharField | `active` / `acknowledged` / `resolved` / `dismissed` |
| `resolved_by_signal` | FK → Signal (nullable) | If this condition was promoted to a Signal |
| `dismissed_reason` | TextField (nullable) | Required when status = dismissed |
| `created_at` | DateTimeField | |
| `resolved_at` | DateTimeField (nullable) | |

**Behavioral rules:**
- Conditions with `status = active` appear on the accountability dashboard
- Dismissing a condition requires a reason (audit trail)
- Conditions resolved by creating a Signal link back to the Signal via `resolved_by_signal`
- Stale conditions (active > N days without acknowledgment) escalate in severity on the dashboard
- The PolicySweepEvaluator de-duplicates: if a condition already exists for the same rule + context, it updates rather than creates a new one

### **4.7 Automation Deferral**

All auto-trigger capabilities are deferred to a future version. The rationale:

- Signal auto-creation, auto-investigation, and auto-escalation are powerful but dangerous if misconfigured.
- Svend will collect longitudinal data on how organizations define and use policies, which thresholds they set, and where manual decisions cluster.
- This data will inform which automations are safe and which are not.
- A research module (separate from production) will analyze policy usage patterns across the ILSSI student population for publication and product development.

---

## **5. CAPA AS COMPLIANCE ARTIFACT**

### **5.1 CAPA Is a Report, Not a Process**

The traditional CAPA process (DRAFT → CONTAINMENT → INVESTIGATION → CORRECTIVE → VERIFICATION → CLOSED) is replaced by the Investigate → Standardize → Verify loop. The CAPA is not the work — it is a generated report that satisfies regulatory requirements.

### **5.2 Report Engine**

The report engine replaces all manual report authoring (CAPA, 8D, A3 summary, management review narrative). The investigator never writes a report. The system assembles it from the atoms of work already done.

**Principle:** Every action in the investigation workspace, every commitment fulfilled, every verification activity completed — these are structured data. The report engine maps this structured data to report sections. The engineer spends time thinking about root causes and countermeasures, not formatting documents.

**5.2.1 Investigation → Report Mapping**

Every investigation stores structured atoms. The report engine queries these atoms and assembles them into sections based on the selected report template.

| Investigation atom | Data type | How it's captured | Report sections it feeds |
|---|---|---|---|
| **Signal source** | FK → Signal | Auto-linked when investigation opens from signal | Problem description, background, scope |
| **Investigation entries** | Ordered list of InvestigationEntry | Investigator adds narrative, data, photos as they work | Chronological investigation record, methodology |
| **Tool outputs** | InvestigationToolLink (generic FK) | Auto-linked when tool runs in investigation context | Analysis results, data tables, charts |
| **Hypotheses** | Synara causal graph nodes | Built interactively in investigation workspace | Root cause analysis, causal chain |
| **Evidence** | Evidence records linked to hypotheses | Auto-created by tool outputs per CANON-002 | Evidence summary, statistical support |
| **Concluded hypothesis** | Hypothesis with P > threshold | Investigator marks conclusion | Root cause statement, confidence level |
| **Commitments** | Commitment records with transition_type | Created at investigation conclusion | Corrective actions, preventive actions, action plan |
| **Fulfilled commitments** | Commitment.status = fulfilled | Marked by owner when work is done | Implementation evidence, completion dates |
| **Mode transitions** | ModeTransition records | Auto-created when commitments fulfilled | Document revisions made, controls added, training assigned |
| **Document revisions** | ControlledDocument version links | Created by revise_document transition | Standard work changes, before/after |
| **Training records** | TrainingRecord + TrainingReflection | Created by train transition, reflections by operators | Training evidence, effectiveness |
| **PC results** | ProcessConfirmation records | Created by verification activities | Effectiveness verification, standard work compliance |
| **Forced failure results** | ForcedFailureTest records | Created by verification activities | Detection verification, control validation |
| **SPC data post-fix** | SPC results with timestamps after fix | Queried by date range relative to fix implementation | Process stability evidence, statistical proof of improvement |
| **Horizontal deployment** | Commitment records for similar processes | Created at investigation conclusion per §14.2 | Horizontal deployment evidence (IATF) |
| **FMIS updates** | FMISRow posterior changes | Auto-updated by evidence from investigation | Risk reduction evidence, before/after RPN |

**5.2.2 Report Templates**

Each template defines: which atoms are required, how they map to sections, what formatting to apply. Templates are org-selectable via QMS Policy.

**ISO 9001 CAPA Template:**

| Section | Source atoms | Required? |
|---|---|---|
| 1. Problem Description | Signal source + investigation description | Yes |
| 2. Immediate Containment | Commitments with early timestamps + containment-tagged entries | If applicable |
| 3. Root Cause Analysis | Concluded hypothesis + evidence chain + methodology (which tools used) | Yes |
| 4. Corrective Action | Mode transitions to Standardize: document revisions, FMIS updates, new controls | Yes |
| 5. Implementation Evidence | Fulfilled commitments with dates + linked artifacts | Yes |
| 6. Effectiveness Verification | PC results + forced failure results + SPC post-fix data | Yes |
| 7. Recurrence Prevention | Ongoing verification schedule (PCs, monitoring, forced failure) | Yes |

**IATF 16949 8D Template (extends ISO 9001):**

| Section | Source atoms | Required? |
|---|---|---|
| D0. Symptom/Emergency Response | Signal source + first investigation entry | Yes |
| D1. Team | Investigation membership with roles | Yes |
| D2. Problem Definition | Is/Is Not analysis from investigation entries (tagged) | Yes |
| D3. Interim Containment | Early commitments + containment evidence | Yes |
| D4. Root Cause | Concluded hypothesis + causal graph visualization | Yes |
| D5. Permanent Corrective Action | Mode transitions + fulfilled commitments | Yes |
| D6. Implementation/Validation | PC results + SPC data + training records | Yes |
| D7. Preventive Action / Horizontal Deployment | Horizontal deployment commitments + similar process evaluation | Yes |
| D8. Team Recognition / Lessons Learned | Investigation conclusion notes + knowledge graph entities created | Yes |

**AS9100D Template (extends ISO 9001):**

| Section | Source atoms | Required? |
|---|---|---|
| RCA Method Documentation | Tool types used (5-why, fishbone, DOE, etc.) from InvestigationToolLink | Yes |
| Risk Assessment | FMIS row before/after + residual risk | Yes |
| Configuration Impact | Linked configuration items affected | If applicable |
| Investigator Qualification | Lead's training records for investigation methodology | Yes (§14.8) |

**FDA 21 CFR 820 Template (extends ISO 9001):**

| Section | Source atoms | Required? |
|---|---|---|
| Statistical Trending | PolicySweepEvaluator trending data for this failure mode category | Yes (mandatory) |
| Risk Analysis per ISO 14971 | FMIS row with severity posterior + risk acceptability per policy | Yes |
| Design History Impact | Linked design files/changes if product-related | If applicable |
| Regulatory Reporting | Complaint linkage, MDR assessment | If applicable |

**5.2.3 Report Assembly Mechanics**

1. **Trigger:** User clicks "Generate Report" on a concluded investigation (or system auto-generates per QMS Policy schedule)
2. **Template selection:** Per QMS Policy `capa.report_standard` — or user selects if multiple are configured
3. **Atom query:** Report engine queries all atoms from the investigation's UUID chain — follows every FK, every mode transition, every commitment
4. **Section assembly:** For each template section, the engine:
   - Collects the mapped atoms
   - Renders them into markdown (narratives stay as-is, tool outputs render as summary + chart reference, evidence renders as structured cards)
   - Flags missing atoms: "Section 6 (Effectiveness Verification): NO PC data found — investigation may not be ready for report generation"
5. **Completeness check:** Report shows a completeness indicator per section. 100% = all required atoms present. <100% = flags what's missing. The user can generate an incomplete report (draft) or wait for atoms to arrive.
6. **Review and publish:** Generated report opens in the document editor (§16.6) for human review. The engineer reviews what the system assembled — corrects emphasis, adds context, approves. Then publishes as a ControlledDocument with e-signature.
7. **Living report:** If new verification data arrives after report generation (more PCs, SPC data), the system can re-generate with updated atoms. The previous version is retained as a revision.

**5.2.4 What the Engineer Does NOT Do**

- Does not write the problem description (auto-populated from signal + investigation)
- Does not list the team members (auto-populated from investigation membership)
- Does not summarize the root cause analysis (auto-populated from concluded hypothesis + evidence)
- Does not list corrective actions (auto-populated from commitments and mode transitions)
- Does not paste evidence (auto-linked from tool outputs)
- Does not track implementation dates (auto-populated from commitment fulfillment timestamps)
- Does not write the effectiveness section (auto-populated from PC/SPC/forced failure data)

**What the engineer DOES do:** Reviews the auto-assembled report for accuracy, adds contextual narrative where the data alone doesn't tell the story, and approves for publication. The 40-60% of quality engineering time currently spent on report writing is reclaimed for investigation and thinking.

### **5.3 Standard Selection**

Organizations select which reporting standard(s) their reports must satisfy. This is a QMS Policy setting:
- **ISO 9001**: standard CAPA fields
- **IATF 16949**: 8D structure with lessons learned, horizontal deployment
- **AS9100D**: RCA method documentation, risk assessment, investigator qualification
- **21 CFR Part 820**: statistical trending (mandatory), risk analysis per ISO 14971
- **Custom**: org-defined template with custom section-to-atom mapping

The system auto-populates what it can and flags what's missing. The report is never a blank form — it always starts from the work already done.

### **5.4 The Chain Beyond CAPA**

Traditional CAPA is a closed record. In Svend, the improvement chain continues:

```
Signal → Investigation → Fix → Standard → Training → Verification
    ↓                                                      ↓
    └──── if recurrence ──── new Signal ←─── gap found ────┘
    ↓
    └──── if pattern across investigations ──→ Kaizen Charter
              (systemic improvement project with its own
               investigation, standards, verification cycle)
```

Kaizen charters are project-level containers that pull in multiple investigations, their evidence, and their commitments. They are the vehicle for systemic improvement — the "we've seen this failure mode 5 times in different processes, the root cause is organizational" response.

**Kaizen Charter specification is deferred to a future revision of this standard.** The mechanism is: a charter links to multiple investigations via UUID, inherits their causal graph nodes, and produces its own commitment chain through the three modes. The pattern-detection that suggests "these investigations should become a charter" is a Synara + knowledge graph capability.

---

## **6. TRAINING AS LOOP PARTICIPANT**

### **6.1 Training Cycle**

```
Document published (Standardize mode)
  → TrainingRequirement auto-created (mode transition §3.2)
  → Roles assigned per document metadata
  → Operator completes training
  → Operator provides hansei reflection:
      - What was clear?
      - What was confusing?
      - What would you change about this instruction?
      - What's different from how you actually do it?
  → Reflections stored as TrainingReflection records
  → Reflections aggregated per document (dashboard view)
  → Pattern detection: if threshold exceeded (QMS Policy §4)
      → system surfaces "revise document" signal
  → Re-training on revised sections only (not full document)
```

### **6.2 Hansei Reflection**

TrainingReflection is a first-class model, not a text field on TrainingRecord.

**Fields:**
- Linked TrainingRecord
- Linked ControlledDocument + version
- Linked ISOSection(s) that were confusing (optional — operator can flag specific sections)
- Free-text reflection (the 4 questions above are prompts, not rigid fields)
- Operator's self-assessed competency level post-training (TWI 0-4)

**Behavioral rules:**
- Reflection is REQUIRED for training completion. A training record without reflection is incomplete.
- Reflections are visible to the document author but NOT to the operator's supervisor in a way that could be punitive. The system separates "what needs to improve in the standard" from "how the operator performed."
- Reflection aggregation is per-document, not per-operator. "4 of 6 operators flagged Section 3 as confusing" is actionable. "Operator X gave negative feedback" is not surfaced.

### **6.3 Competency Model (TWI Job Instruction)**

- Level 0: None
- Level 1: Awareness (has seen the standard)
- Level 2: Supervised (can perform under guidance)
- Level 3: Competent (can perform independently)
- Level 4: Trainer (can teach others)

Competency level advances through: training (0→1), supervised practice + PC pass (1→2), sustained PC pass rate (2→3), demonstrated teaching ability (3→4). Each transition requires evidence, not manager override.

---

## **7. VERIFICATION ACTIVITIES**

### **7.1 Process Confirmations (PCs)**

A Process Confirmation is a structured gemba observation that answers two questions:
1. Was the standard followed?
2. Did following the standard produce the expected outcome?

**Diagnostic matrix:**

| Standard followed? | Outcome correct? | Diagnosis | System response |
|---|---|---|---|
| Yes | Yes | System works | Score contributes to operator/process confidence |
| No | — | Standard unclear, impractical, or training gap | Surfaces `revise_document` or `train` signal per QMS Policy |
| Yes | No | Process design is broken | Surfaces `investigate` signal — HIGH VALUE, standard faithfully encodes a failed process |

**PC Model:**
- Linked ControlledDocument (the standard being confirmed) — REQUIRED
- Linked Employee (operator observed) — REQUIRED
- Linked process area / zone
- Observer (the person conducting the PC) — REQUIRED
- Observation items derived from the document's key steps
- Each item: followed (yes/no/NA) + outcome (pass/fail/NA) + notes
- Overall result + observer notes + improvement observations
- Photos (linked to specific steps)

**Rules of engagement (adapted from STOP methodology):**
- ALWAYS interact with the operator — this is not surveillance
- Observe objectively — record what you see, not what you expect
- Don't lead or question the operator during the task
- Always explain what you're doing and why before starting
- Always look for what's going RIGHT first — acknowledge good practice before addressing gaps

**PC Threshold Mechanics (governed by QMS Policy §4):**
- PC results are per-operator, per-standard
- Trailing window (default 5 PCs, org-configurable)
- Pass rate below threshold surfaces a signal (NOT auto-creates retraining)
- Cooldown: same operator+standard signal cannot fire within cooldown_days
- Escalation: if N operators fail the same standard within window → surfaces `revise_document` signal instead of N individual training signals
- PCs continue during active retraining — no coverage gap
- In-flight retraining is visible on accountability dashboard with duration tracked against org-defined policy

### **7.2 Forced Failure Tests**

A Forced Failure test deliberately creates conditions where the process should fail, then observes whether detection controls catch it.

**Two modes:**
1. **Hypothesis-driven**: linked to specific FMEARow. Tests a specific detection control against a specific failure mode.
2. **Exploratory**: not linked to a specific row. Probes boundary conditions to discover unknown failure modes.

**ForcedFailureTest Model:**
- Linked FMEARow (for hypothesis-driven) — nullable for exploratory
- Test plan: conditions to create, expected detection response
- Safety review: boolean confirmation that test can be conducted safely — REQUIRED before execution
- Control being tested: text description + linked control artifact if applicable
- Result: detected / not_detected / partially_detected
- Detection count: integer — how many injected failures were detected
- Injection count: integer — how many failures were injected
- Evidence: photos, data, measurements
- Temporal state: when the test was conducted (for Sunrise tracking §8)

**Behavioral rules:**
- Forced failure results feed directly into FMEA detection scoring (§8.3)
- A `not_detected` result on a hypothesis-driven test auto-surfaces an `investigate` signal — detection gaps are ALWAYS investigated
- Exploratory test results that discover new failure modes surface an `add_control` signal
- Safety review is a hard gate — the system will not accept results without safety confirmation

---

## **8. FMEA REFORMATION — BAYESIAN S/O/D**

### **8.1 Thesis**

The AIAG/VDA FMEA scoring system asks committees to vote on integers. Svend replaces this with computed values from data, using the Sunrise Problem (sequential Bayesian updating) as the unified mathematical framework.

All three dimensions — Severity, Occurrence, Detection — are posterior distributions updated by evidence. The system displays the posterior mean and credible interval alongside the traditional 1-10 integer for backward compatibility.

### **8.2 Operational Definitions**

FMEA rows MUST contain operational definitions for key terms. An operational definition is a link to a schematic entity in the knowledge graph — not free text.

**Why:** "Temperature too high" means nothing without: which temperature? Measured how? At what point in the process? What threshold defines "too high"? The operational definition links the FMEA term to a specific measurable quantity with units, measurement method, and linked equipment.

**Enforcement:** Operational definitions are developed inline as the user adds FMEA rows. Not all at once. The system tracks which terms are defined and which are not, and surfaces undefined terms as knowledge gaps in the Dynamic Process Model (§9).

### **8.3 Detection — Sunrise Problem Over Forced Failure Results**

Detection is the probability that the control system catches a failure when it occurs.

**Model:** Beta-Binomial. Prior: Beta(1, 1) (uninformative). Updated by each forced failure test.

**Updating rule:** If test detects `d` of `n` injected failures:
- Posterior: Beta(α + d, β + (n - d))
- Detection rate estimate: posterior mean = (α + d) / (α + β + n)

**Temporal/factorial state:** Each forced failure test records:
- Which control was being tested
- When the test was conducted
- What conditions were present

This allows decomposition: "Under Control A, detection is Beta(5, 2). Under Control B, detection is Beta(4, 2)." You can directly compare controls by comparing their posterior distributions.

**FMEA detection score:** The posterior mean maps to the traditional 1-10 scale for backward compatibility. But the system displays the actual posterior distribution and credible interval. The integer is a lossy compression for organizations that need it.

### **8.4 Occurrence — Sunrise Problem Over Process Data**

Occurrence is the rate at which this failure mode manifests.

**Model:** Beta-Binomial over production data. Prior: from historical failure rate or Beta(1, 1) if no history.

**Updating rule:** Over a production window, `f` failures observed in `n` units:
- Posterior: Beta(α + f, β + (n - f))
- Occurrence rate estimate: posterior mean

**Integration with SPC:** SPC violations that correspond to this failure mode contribute occurrence evidence. If an SPC signal is classified (via the knowledge graph) as "this failure mode occurring," it increments the failure count.

**Integration with Cpk:** Process capability provides a distributional view of occurrence. If the process distribution tail overlaps with the failure region, the expected occurrence rate can be computed directly from the capability analysis. This is complementary to the counting approach — the counting approach is empirical, the Cpk approach is model-based.

### **8.5 Severity — Categorical-Dirichlet Over Consequence Evidence**

Severity is the impact of the failure mode on the customer, downstream process, or safety. Unlike detection and occurrence, severity is inherently subjective — an operator who witnessed a near-miss has information no sensor captured. The categorical model lets human judgment count alongside data-driven observations.

**Model:** Categorical-Dirichlet. Five severity categories with a Dirichlet prior.

**Severity categories:**

| Category | Label | Examples | AIAG mapping |
|---|---|---|---|
| 1 | Negligible | No effect on product/process, cosmetic only | 1-2 |
| 2 | Minor | Minor rework, no customer impact, caught in-process | 3-4 |
| 3 | Moderate | Significant rework/scrap, potential customer dissatisfaction | 5-6 |
| 4 | Severe | Product fails in field, line stoppage, regulatory exposure | 7-8 |
| 5 | Catastrophic | Safety hazard, recall, injury potential | 9-10 |

**Prior:** Dirichlet(1, 1, 1, 1, 1) — uninformative. Each category equally likely before evidence.

**Updating rule:** Each consequence observation is classified into one of the 5 categories. The Dirichlet posterior updates:
- Posterior: Dirichlet(α₁ + n₁, α₂ + n₂, α₃ + n₃, α₄ + n₄, α₅ + n₅) where nₖ is the count of observations in category k.

**Sources of consequence evidence:**
- Investigation records that touched this failure mode: investigator classifies consequence severity
- Customer complaints linked to this failure mode: complaint severity classification
- NCR records: disposition severity (scrap, rework, sort, return)
- Field returns or warranty claims: classified by field engineer

**Displayed severity:** Org-configurable via QMS Policy:
- `severity.display = expected_value` — Σ(category × P(category)). Default.
- `severity.display = modal` — most probable category.
- `severity.display = percentile_90` — 90th percentile for risk-averse organizations (aerospace, medical).

**When no data exists:** The system displays "no empirical severity data — committee estimate in effect" and retains the manually-entered score. The Dirichlet posterior is not computed until at least one consequence observation exists. The system does NOT fabricate data.

### **8.5.1 Unified Scoring Framework Summary**

| Dimension | Model | Prior | Update mechanism | Evidence source |
|---|---|---|---|---|
| **Detection** | Beta-Binomial | Beta(1,1) | d detected / n injected per forced failure test | ForcedFailureTest results |
| **Occurrence** | Beta-Binomial | Beta(1,1) | f failures / n units per production window | Process data, SPC |
| **Severity** | Categorical-Dirichlet | Dir(1,1,1,1,1) | Each consequence classified into 5 categories | Investigations, NCRs, complaints, field returns |

### **8.6 The SVEND FMEA Model**

The reformed FMEA supersedes the AIAG 4th Edition methodology by integrating its requirements into a data-driven system. Organizations opt into the level of rigor via QMS Policy (§4):

| Policy setting | Behavior |
|---|---|
| `fmea.methodology = aiag_4th` | Traditional 1-10 manual scoring. Forced failure and process data ignored for scoring. |
| `fmea.methodology = svend_bayesian` | Bayesian S/O/D with posterior distributions. Manual scores used as priors, data updates them. |
| `fmea.methodology = svend_full` | Full integration: operational definitions required, forced failure mandatory for detection, process data mandatory for occurrence, consequence data for severity. |

This allows organizations to adopt progressively. A small shop starts with AIAG 4th. As they use Svend's investigation and verification tools, data accumulates and the Bayesian model becomes useful. They switch when ready.

### **8.7 FMIS Model Schema**

FMIS (Failure Modes Investigation System) is the investigation-native FMEA. It coexists with the standalone AIAG FMEA tool (QMS-001 §4.1). The standalone tool is a calculator. FMIS participates in the Investigate → Standardize → Verify loop.

**FMISRow fields:**

| Field | Type | Purpose |
|---|---|---|
| `id` | UUID PK | Standard identifier |
| `fmis` | FK → FMIS | Parent FMIS document |
| `investigation` | FK → Investigation | The investigation that created or last modified this row |
| `process_model` | FK → ProcessModel (nullable) | Which process model this row's entities belong to |
| | | |
| **Failure mode** | | |
| `failure_mode_entity` | FK → KG Entity (nullable) | Operational definition — links to knowledge graph entity. Null = not yet defined (knowledge gap). |
| `failure_mode_text` | CharField | Human-readable label. May be derived from entity or manually entered. |
| **Effect** | | |
| `effect_entity` | FK → KG Entity (nullable) | Operational definition of the downstream effect |
| `effect_text` | CharField | Human-readable label |
| **Cause** | | |
| `cause_entity` | FK → KG Entity (nullable) | Operational definition of the cause mechanism |
| `cause_text` | CharField | Human-readable label |
| | | |
| **Controls** | | |
| `prevention_control` | TextField | Current prevention control description |
| `detection_control` | TextField | Current detection control description |
| | | |
| **Severity (Categorical-Dirichlet §8.5)** | | |
| `severity_alpha` | JSONField | Dirichlet alpha vector [α₁, α₂, α₃, α₄, α₅]. Default: [1,1,1,1,1] |
| `severity_observation_count` | IntegerField | Total consequence observations across all categories |
| `severity_manual` | IntegerField (1-10, nullable) | Manual override / initial committee estimate. Used when `severity_method = manual`. |
| `severity_method` | CharField | `manual` / `bayesian`. Determines which value is displayed. |
| | | |
| **Occurrence (Beta-Binomial §8.4)** | | |
| `occurrence_alpha` | FloatField (default=1) | Beta prior α |
| `occurrence_beta` | FloatField (default=1) | Beta prior β |
| `occurrence_failures` | IntegerField (default=0) | Cumulative failure count |
| `occurrence_units` | IntegerField (default=0) | Cumulative units observed |
| `occurrence_manual` | IntegerField (1-10, nullable) | Manual override |
| `occurrence_method` | CharField | `manual` / `bayesian` |
| | | |
| **Detection (Beta-Binomial §8.3)** | | |
| `detection_alpha` | FloatField (default=1) | Beta prior α |
| `detection_beta` | FloatField (default=1) | Beta prior β |
| `detection_detected` | IntegerField (default=0) | Cumulative detections across all forced failure tests |
| `detection_injected` | IntegerField (default=0) | Cumulative injections across all forced failure tests |
| `detection_manual` | IntegerField (1-10, nullable) | Manual override |
| `detection_method` | CharField | `manual` / `bayesian` |
| | | |
| **Computed** | | |
| `rpn` | IntegerField (computed) | S × O × D using posterior means mapped to 1-10 (or manual values, depending on method) |
| `last_evidence_date` | DateTimeField (nullable) | When posteriors were last updated from evidence |
| | | |
| **Temporal state** | | |
| `created_at` | DateTimeField | Row creation |
| `updated_at` | DateTimeField | Last modification |

**Operational definition enforcement:**

- Entity FKs (`failure_mode_entity`, `effect_entity`, `cause_entity`) are nullable. An FMIS row with null entities is valid but flagged as having undefined terms — a knowledge gap.
- When `fmis.methodology = svend_full` (QMS Policy), the system requires all three entities to be linked before the row can participate in simulation (§9).
- Entities are developed inline as the user adds rows. Not front-loaded. The system tracks definition completeness: "3 of 7 failure modes have operational definitions."
- Entity links enforce semantic consistency. If two rows reference the same `cause_entity`, they share the same operational definition. No semantic drift.

**Forced failure test linkage:**

ForcedFailureTest records (§7.2) link to FMISRow. When a test concludes:
1. `detection_detected += test.detection_count`
2. `detection_injected += test.injection_count`
3. `detection_alpha` and `detection_beta` recomputed from cumulative counts
4. `last_evidence_date` updated
5. If multiple controls tested, each test records which control was active — the row tracks aggregate detection, but individual test records preserve the factorial state for control comparison.

**Migration from standalone FMEA:**

An existing FMEARow can be linked to an FMISRow via `fmea_row` FK (nullable). This allows:
- Importing manual S/O/D scores as Bayesian priors
- Preserving existing RPN history
- Progressive migration: start with AIAG, switch to Bayesian as evidence accumulates

---

## **9. DYNAMIC PROCESS MODEL**

### **9.1 Purpose**

The Dynamic Process Model is a quantitative knowledge graph that represents what the organization KNOWS about a process — and, critically, what it does NOT know.

It is not a digital twin. It is not a physics simulation. It is a collection of calibrated local relationships (X causes Y with measured effect size and confidence) that converge toward a representation of real-world process behavior as investigations, DOEs, and verification activities add data.

### **9.2 Schema**

The process model shares its schema with the knowledge graph. Entities and relationships are typed:

**Entity types:**
- `controllable_input`: a process parameter the operator can change (temperature, speed, pressure)
- `noise_input`: a process parameter that varies but cannot be directly controlled (ambient humidity, material lot variation)
- `intermediate`: a measurable quantity between input and output (viscosity, surface roughness)
- `output`: the final quality characteristic (dimension, strength, appearance)
- `specification`: the acceptable range for an output (USL, LSL, target)

**Each entity carries:**
- Operational definition (units, measurement method, linked equipment)
- Current distribution (mean, std, shape — from SPC or process data)
- Linked FMEA rows where this entity appears

**Relationship types:**
- `causal`: X causes Y. Has: effect_size, confidence_interval, source (investigation/DOE UUID), calibration_date
- `correlational`: X and Y move together but causation not established. Has: correlation coefficient, source
- `confounded`: X appears to cause Y but a known confounder exists. Has: confounder entity link

### **9.3 Building the Model**

The user builds the process model skeleton, not the system.

**Starting point:** FMEA rows for the process. Each row already asserts: "failure mode X is caused by mechanism Y affecting output Z." The system proposes a causal skeleton from the FMEA. The user confirms, modifies, adds entities.

**Progressive enrichment:**

| Activity | What it contributes to the model |
|---|---|
| Investigation with DOE | Calibrated causal edge: effect size ± CI, from controlled experiment |
| Investigation without DOE | Observational edge: weaker evidence, wider CI |
| Process Confirmation | Confirms or challenges expected behavior (model validation) |
| Forced Failure Test | Empirical detection probability for specific failure mode path |
| SPC data | Ongoing process variability estimates for entity distributions |
| Gage R&R | Measurement system capability — determines whether edges are trustworthy |

### **9.4 Gap Exposure**

The model's primary value is exposing what you don't know:

- **Uncalibrated edges**: "We assert temperature affects viscosity but have no measured effect size." The system flags these as investigation opportunities.
- **Stale edges**: "This relationship was calibrated 8 months ago. The process has changed since (SPC signals detected)." The system flags these for re-investigation.
- **Missing entities**: "FMEA lists 'material hardness' as a cause but we have no measurement system for it." The system flags this as a measurement gap.
- **Conflicting evidence**: "Two investigations measured the same relationship and got different effect sizes." The system flags this as a resolution opportunity.

### **9.5 Model Integrity**

- **No drift tolerated**: if SPC detects a process shift that contradicts a model edge, the edge is flagged as potentially stale. The system does NOT auto-update the edge — it surfaces a signal.
- **Every edge has provenance**: investigation UUID, DOE UUID, or "manual estimate (no empirical data)." Auditors can trace any model assertion to its source.
- **The model is lossy by design**: it does not attempt to capture all process dynamics. Missing edges are visible. The user decides where to invest investigation effort to reduce loss.

### **9.6 Investigation as Process Model Subset**

An investigation does not operate on its own isolated graph. It operates on a **subset** of the process model, scoped by:
- **Time window**: the period under investigation (e.g., "production from Jan-Mar 2026")
- **Condition set**: the specific factors and outputs relevant to the problem (e.g., "Zone 3 temperature, viscosity, defect rate")
- **Entity scope**: which nodes and edges from the process model are included

The investigation's Synara causal graph IS a subgraph of the process model. When the investigation concludes:
1. New calibrated edges (from DOE, statistical analysis) are proposed as updates to the process model
2. The investigator confirms which relationships should persist in the model
3. Confirmed edges update the process model with new effect sizes, confidence intervals, and provenance (investigation UUID)
4. Contradicting edges (new data conflicts with existing model) are flagged for review — the system does not silently overwrite

This means: you never analyze the entire process model. You subset it to what's relevant. The model grows through investigation conclusions, not through bulk ingestion.

### **9.7 Simulation**

When the model has sufficient calibrated edges, the user can run simulations:

- **Parameter sweep**: "What if input X changes from 10 to 15?" → Monte Carlo propagation through calibrated edges → output distribution
- **Failure injection**: "What if this input goes out of spec?" → model shows which outputs are affected and by how much
- **Sensitivity analysis**: "Which inputs matter most for this output?" → rank by influence (partial derivatives or variance decomposition)
- **Counterfactual**: "After implementing fix, run model WITHOUT the fix" → compare to post-fix actuals → real effect size

The user orchestrates. The system computes. The system surfaces when a simulation result depends on an uncalibrated or stale edge — "this prediction relies on an edge with no empirical data."

---

## **10. CI READINESS SCORE**

### **10.1 Purpose**

The CI Readiness Score answers: "Is the loop turning?" It is NOT a compliance score. It measures improvement system health.

### **10.2 Indicators**

| Indicator | Question | Data source | Absence penalty |
|---|---|---|---|
| Signal detection rate | Are problems found internally before customers find them? | Internal signals vs. customer complaints ratio | No signals = 0 (system is blind) |
| Investigation velocity | Are investigations producing conclusions? | Close rate, median time, % with evidence | No investigations = 0 |
| Hypothesis testing rate | Are conclusions tested, not assumed? | % of investigations with DOE or statistical evidence | No testing = heavy penalty |
| Standardization lag | How fast do fixes become standards? | Investigation conclude → document publish time | Concluded investigations with no document revision = penalty |
| Training coverage | Are people current on current standards? | % trained on latest version, % with reflections | No training records = 0 |
| Verification activity | Are PCs and forced failures happening? | Schedule adherence vs. QMS Policy defined frequency | No verification = 0 |
| Recurrence rate | Are fixed problems staying fixed? | Repeat failure modes across investigations | Recurrence = heavy penalty |
| Standard work quality | Are standards followed AND producing results? | PC pass rates (followed × outcome) | No PCs = unknown |
| Detection capability | Do controls catch failures? | Forced failure detection rates vs. FMEA detection scores | No forced failures = unknown (not penalized, but flagged) |
| Commitment fulfillment | Are people doing what they committed to? | On-time rate, broken commitment patterns | No commitments = system not in use |

### **10.3 Scoring Philosophy**

- Absence = failure. No data means the system is not running, which is worse than bad data.
- Evidence decays. A half-life function reduces the weight of old evidence. The decay rate is org-configurable via QMS Policy.
- Overdue items are worst. An overdue commitment is worse than a late commitment. The system knew and didn't act.
- The score CAN go down when you add data. Discovering a problem is progress. Hiding it is not.
- The score formula, weights, and thresholds are defined in QMS Policy. This standard defines the INDICATORS, not the weights. Different industries weight differently.

---

## **11. AUDITOR PORTAL**

### **11.1 Purpose**

Read-only view of system data organized for external auditors. Same data as the internal system, different lens.

### **11.2 Access**

Time-limited token (ActionToken pattern). Quality manager generates a link, shares with auditor. No account required. Token expires after org-defined period.

### **11.3 Views**

- **Clause-organized evidence**: select standard (ISO 9001, IATF 16949, AS9100D, ISO 13485), see relevant records per clause
- **Traceability chains**: click any record → see the full chain: signal → investigation → fix → document → training → verification
- **Generated CAPA reports** (§5): pre-assembled compliance artifacts
- **Readiness score trend**: shows improvement over time
- **Policy documentation**: org-defined policies that were in effect at any point in time

---

## **12. BIAS DETECTION AND ACCOUNTABILITY TRANSPARENCY**

### **12.1 Purpose**

The system surfaces patterns that suggest bias in quality decisions. It does not accuse — it makes data visible.

### **12.2 Patterns Monitored**

| Pattern | What the system surfaces |
|---|---|
| Retraining concentration | "Supervisor A triggers retraining 3x more frequently than peer supervisors for comparable PC scores" |
| Investigation assignment bias | "Investigations involving Process X are always assigned to Person Y — no rotation" |
| Commitment fulfillment disparity | "Team A fulfills 90% of commitments on time. Team B fulfills 40%. Both have similar workloads." |
| Severity scoring inconsistency | "FMEA reviews by Reviewer X consistently score severity 2 points lower than empirical consequence data suggests" |
| Signal dismissal rate | "Site A dismisses 60% of signals. Site B dismisses 10%. Comparable process complexity." |

### **12.3 Behavioral Rules**

- Bias detection outputs are visible to quality management, not to the individuals being compared.
- The system presents data, not conclusions. "These numbers are different" — not "this person is biased."
- Bias patterns that persist are surfaced as signals that may warrant investigation (using the same Signal mechanism §3.1).
- The bias detection service is global and operates across all QMS data, not scoped to a single module.

---

## **13. ADVANCED PROCESS CONTROL INTEGRATION**

The five APC Frontiers (see `docs/planning/ADVANCED_PROCESS_CONTROL.md` for research code and derivations) extend the Verify and Signal modes with capabilities no competitor can replicate. They are not separate features — they are the mathematical foundation that makes the loop world-class.

### **13.1 Frontier 1: Control Charts as Information Channels**

**Mode:** Verify (SPC monitoring)
**Integration point:** Process Model (§9) + SPC engine

Reframes control chart design from "minimize ARL" to "maximize mutual information between process state and chart signal." Shannon (1948) and Shewhart (1931) unified.

**What the system gains:**
- Optimal chart design per process: given process noise, shift magnitude, and subgroup strategy, compute which chart type (Xbar, EWMA, CUSUM) transmits the most information per sample
- i-type Cpk: capability measured as differential entropy — connects process capability directly to information theory
- Data Processing Inequality applied to subgrouping: quantify how much information rational subgrouping destroys, enabling informed subgroup strategy selection

**Integration with §9 (Process Model):** The process model stores entity distributions (mean, std). The information-theoretic chart optimizer uses these distributions to recommend chart parameters. As the process model refines (more investigation data), chart recommendations update.

### **13.2 Frontier 2: Reaction Plan Stability**

**Mode:** Verify (new verification activity type)
**Integration point:** Commitment system (§3.3) + QMS Policy (§4)

Models the SPC feedback loop (process → chart → operator → reaction → process) as a discrete-time control system. Applies Nyquist stability criteria to detect when a reaction plan will cause tampering (Deming funnel).

**What the system gains:**
- **Stability certificate**: before a reaction plan is deployed (as a Commitment from an investigation), the system can assess whether the plan's loop gain will destabilize the process
- **Tampering detection**: if SPC data shows increasing variance after a reaction plan is implemented, the system can diagnose whether the operator is over-adjusting (loop gain > 1)
- Reaction plan verification becomes a Verify mode activity alongside PCs and forced failures

**Behavioral rule:** When a Commitment of type `monitor` creates an SPC configuration with an associated reaction plan, the system computes estimated loop gain from the plan parameters (reaction magnitude, delay, process dynamics). If estimated gain > stability threshold → PolicyCondition surfaced: "Reaction plan may destabilize process."

### **13.3 Frontier 3: Capability as Distribution Over Distributions**

**Mode:** Verify (SPC monitoring) + Signal (capability degradation)
**Integration point:** Process Model (§9) + FMIS occurrence (§8.4)

Treats Cpk not as a point estimate but as a trajectory in distribution space. Puts a control chart ON Cpk itself, using Wasserstein distance or JSD as the measurement.

**What the system gains:**
- **Cpk control chart**: EWMA/CUSUM on the Cpk time series. Detects capability degradation before individual measurements trigger OOC signals.
- **Distributional drift decomposition**: when Cpk changes, the system identifies WHETHER it's a location shift, scale change, or shape change — each has different assignable causes.
- **Cpk degradation as Signal source**: when the Cpk control chart triggers, the system surfaces a PolicyCondition. This is a leading indicator — the process is getting worse before it produces defects.

**Integration with §8.4 (FMIS Occurrence):** Cpk data feeds the occurrence posterior. A declining Cpk trend increases the expected occurrence rate for related FMIS rows.

### **13.4 Frontier 4: Covariate-Adjusted Charts**

**Mode:** Verify (SPC monitoring)
**Integration point:** Process Model (§9)

Replaces rational subgrouping for heterogeneous processes (high-mix/low-volume, job shops, continuous processes) with regression-adjusted control charts.

**What the system gains:**
- **Covariate-adjusted monitoring**: regress out known sources of between-unit variation (material lot, operator, tool wear), chart the residuals. A shift in residuals = assignable cause, not just process heterogeneity.
- **Mixed-effects SPC**: hierarchical model where variation is partitioned into machine, shift, operator, and residual components. Control chart on the appropriate component.

**Integration with §9 (Process Model):** The process model's causal edges define which covariates to adjust for. If the model says "material hardness affects output with β=0.8," the covariate-adjusted chart regresses out hardness before monitoring. As the model learns new relationships, chart adjustments update.

**Integration with §8.7 (FMIS):** FMIS operational definitions (entity FKs) define what the covariates ARE. The same entities that appear in the FMIS cause column appear as covariates in the adjusted chart.

### **13.5 Frontier 5: Detection/Diagnosis Unification (Bayesian Fault Classifier)**

**Mode:** Signal generation + Verify (real-time monitoring)
**Integration point:** FMIS (§8) + SPC engine + Synara belief engine

The most architecturally significant frontier. Unifies SPC detection and RCA diagnosis into a single operation.

**Core idea:** Maintain posterior probabilities P(fault_k | data) over a library of fault modes (from FMIS), updated sequentially with each new SPC observation. When the posterior on any fault exceeds a threshold, that IS the signal — and it comes with a diagnosis attached.

**What the system gains:**
- **Detection + diagnosis simultaneously**: instead of "OOC signal detected → open investigation → figure out what went wrong," the system says "OOC signal detected, P=0.85 this is FMIS row #47 (tool wear causing dimensional shift)."
- **FMIS priors → fault classifier**: FMIS occurrence scores become prior probabilities for each fault mode. High-occurrence faults get higher priors. Forced failure detection rates inform the likelihood model.
- **Faster loop closure**: the investigation starts with a ranked hypothesis list, not a blank slate. The investigator confirms or rejects the system's top hypothesis with evidence.
- **Western Electric rules replaced**: the 8 classical run rules are primitive pattern classifiers. Each rule loosely corresponds to a fault type (trend = gradual degradation, oscillation = overcorrection, etc.). The Bayesian fault classifier is the principled generalization — it detects the same patterns with proper uncertainty quantification.

**Architecture:**

```
FMIS fault library (failure modes with occurrence priors)
    ↓ priors
BayesianFaultClassifier
    ↑ observations (SPC data stream)
    ↓ posteriors P(fault_k | x_1:t)
    ↓
Signal: "fault_k detected with P=0.85"
    + ranked hypothesis list for investigation
```

**Key finding from APC research:** The EWMA forgetting factor λ and the Bayesian classifier forgetting factor α are the same mathematical object — exponential forgetting at different levels of the inference hierarchy. Optimal rate ≈ 1/τ where τ is mean time between fault transitions. Frontiers 1 and 5 are the same optimization problem viewed from different levels.

**Integration with §3.1 (Signals):** The fault classifier's output is a PolicyCondition (§4.6), not an auto-created Signal. The system surfaces: "Bayesian fault classifier indicates FMIS row #47 with P=0.85." A human reviews and creates the Signal if warranted. This preserves HITL (§1.3 constraint #2).

---

## **14. INDUSTRY-SPECIFIC REQUIREMENTS**

LOOP-001's core architecture (three modes, three mechanisms) is industry-agnostic. Industry-specific requirements are handled through QMS Policy (§4) and optional capability modules. This section specifies what each tier requires beyond the core.

### **14.1 Requirements Matrix**

| Capability | ISO 9001 | IATF 16949 | AS9100D | FDA/13485 | LOOP-001 section |
|---|---|---|---|---|---|
| **Core loop (Investigate → Standardize → Verify)** | Required | Required | Required | Required | §2 |
| **Signals + Commitments + Mode Transitions** | Required | Required | Required | Required | §3 |
| **QMS Policy service** | Optional | Required | Required | Required | §4 |
| **Generated CAPA report** | Required | Required (8D format) | Required (8D + AS13100) | Required (statistical trending) | §5 |
| **Training with hansei reflection** | Good practice | Required | Required | Required | §6 |
| **Process Confirmations** | Good practice | Required (LPA) | Good practice | Good practice | §7.1 |
| **Forced Failure Tests** | Not required | Required (error-proofing) | Recommended | Required (process validation) | §7.2 |
| **FMIS (Bayesian FMEA)** | Optional | Recommended | Recommended | Required (ISO 14971) | §8 |
| **Dynamic Process Model** | Optional | Optional | Optional | Optional | §9 |
| **CI Readiness Score** | Optional | Recommended | Recommended | Recommended | §10 |
| **Auditor Portal** | Good practice | Good practice | Good practice | Good practice | §11 |
| **Horizontal deployment** | Not required | **Required** | Implied | Good practice | §14.2 |
| **Error-proofing device tracking** | Not required | **Required** | Not required | Not required | §14.3 |
| **Configuration management** | Not required | Implied | **Required** | **Required** (design history) | §14.4 |
| **First Article Inspection** | Not required | Recommended | **Required** (AS9102) | Not required | §14.5 |
| **Statistical CAPA trending** | Not required | SPC required | Not required | **Required** (mandatory) | §14.6 |
| **Post-market surveillance** | Not required | Warranty analysis | Not required | **Required** | §14.7 |
| **Investigator qualification** | Not required | Competency required | **Required** (AS13100 8D) | Cross-functional required | §14.8 |
| **APC Frontiers** | Optional | Recommended | Recommended | Optional | §13 |

### **14.2 Horizontal Deployment**

**Required by:** IATF 16949 §10.2.3
**When:** Every investigation conclusion that produces a corrective action.

Before an investigation can transition from Investigate to Standardize, the system surfaces similar processes and products from the knowledge graph: "This failure mode was found in Process A. Processes B, C, D share the same cause entity. Has this fix been evaluated for those processes?"

**Behavioral rules:**
- The horizontal deployment check is a REQUIRED step in investigation conclusion when QMS Policy `investigation.horizontal_deployment = required` (default for IATF).
- The investigator must either: (a) create Commitments to evaluate the fix in similar processes, or (b) document why horizontal deployment is not applicable, with a reason.
- Horizontal deployment evaluation results are tracked — did the fix apply? Was a new investigation needed?
- CAPA report generation (§5) includes horizontal deployment evidence when the org's policy requires it.

### **14.3 Error-Proofing Device Tracking**

**Required by:** IATF 16949 §10.2.4

Error-proofing (poka-yoke) devices require their own tracking, separate from MeasurementEquipment:

**ErrorProofingDevice model:**
- Name, type, location, process step
- Linked FMIS row (which failure mode does this device prevent/detect?)
- Linked control plan entry
- Test frequency (per QMS Policy)
- Challenge part requirements (part number, storage, calibration)
- Failure response plan: what happens when the device malfunctions?
- Test records: date, result (pass/fail), tester, challenge part used

**Behavioral rules:**
- Overdue device tests surface as PolicyConditions (§4.6)
- Device test failure auto-surfaces a `forced_failure` Signal (detection gap confirmed)
- Device records linked to FMIS rows — a device failure updates the detection evidence for the corresponding failure mode

### **14.4 Configuration Management**

**Required by:** AS9100D (product lifecycle), FDA/ISO 13485 (design history)

Configuration management tracks which version of which component is in which assembly at which lifecycle stage.

**Integration with LOOP-001:**
- Configuration changes are Signal sources (§3.1): when a product configuration changes, the system evaluates whether linked FMIS rows, control plans, and standards need review.
- Design History File (DHF/DDF under QMSR) is a view on the investigation + document revision chain for a specific product configuration.
- Configuration items link to FMIS entities — the same operational definitions used in failure mode analysis identify the configuration items.

**Model specification deferred** — configuration management is a deep domain that deserves its own standard (CONFIG-001). LOOP-001 specifies the integration points: configuration changes produce Signals, configuration items link to FMIS entities and process model nodes.

### **14.5 First Article Inspection (FAI)**

**Required by:** AS9100D §8.5.1.3, AS9102

FAI is a Verify mode activity that confirms a production process can produce conforming parts.

**FAI model:**
- Linked part number / configuration item
- Three forms: Part Number Accountability (Form 1), Product Accountability (Form 2), Characteristic Accountability (Form 3)
- Inspector (must be independent from manufacturing — enforced)
- Linked measurement equipment (must be calibrated)
- Key characteristics identified and verified (per AS9103)
- Result: approved / conditionally approved / not approved
- Re-trigger conditions: any change affecting fit, form, or function

**Behavioral rules:**
- FAI is a verification activity that can be scheduled via Commitment (type: `first_article_inspection`)
- Configuration or design changes on the linked part auto-surface a PolicyCondition: "FAI may need to be repeated due to [change description]"
- FAI results feed back into FMIS: if FAI fails on a characteristic linked to an FMIS row, occurrence evidence is updated

### **14.6 Statistical CAPA Trending**

**Required by:** FDA 21 CFR 820.100 (mandatory statistical methods)

The system must provide statistical trend analysis on CAPA/investigation data as a built-in capability, not a manual export-and-analyze process.

**What the system computes (PolicySweepEvaluator, §4.6.3):**
- Pareto analysis of investigation root cause categories (top N causes over trailing period)
- Run chart / control chart on investigation creation rate by category
- Recurrence detection: same failure mode or cause appearing across N investigations within window
- Complaint-to-investigation conversion rate trending
- Time-to-closure trending (are investigations getting faster or slower?)

**Behavioral rules:**
- Statistical trending results are included in management review auto-snapshot
- Trending evidence is included in generated CAPA reports when org policy is `capa.include_trending = true` (default for FDA-regulated organizations)
- Trend analysis uses the same SPC engine (control charts on count data) that monitors process variables — the QMS monitors itself

### **14.7 Post-Market Surveillance and Field Feedback**

**Required by:** ISO 13485 (PMS), FDA (MDR/vigilance), IATF 16949 (warranty/NTF)

Field data must flow back into the loop as Signals that update FMIS, the process model, and the risk register.

**Capabilities:**
- **Complaint → Signal pipeline**: customer complaints (already a Signal source, §3.1) with severity classification, regulatory reporting timeline tracking, and investigation linkage
- **Adverse event / vigilance**: regulatory reporting deadlines tracked per QMS Policy. Overdue reports surface as critical PolicyConditions.
- **NTF (No Trouble Found) analysis**: warranty returns where no defect is found require structured analysis. NTF patterns are themselves a signal — they may indicate a customer-use problem, intermittent failure, or inadequate test coverage.
- **Field data → FMIS feedback**: field failures update FMIS occurrence and severity posteriors. A field failure on a failure mode previously rated "negligible" is a high-value severity evidence update.
- **Field data → process model feedback**: field performance data updates process model output distributions. Drift between production data and field data is a signal.

### **14.8 Investigator Qualification**

**Required by:** AS13100 (trained 8D practitioner), IATF (competent problem solvers), FDA (cross-functional teams)

**Behavioral rules (enforced via QMS Policy):**
- `investigation.lead_qualification = required` — investigation lead must have a TrainingRecord at competency level ≥ 3 on the applicable investigation methodology (8D, A3, RCA)
- `investigation.team_composition = cross_functional` — investigation membership must include at least N distinct roles/departments (configurable, default 2)
- Qualification requirements are checked at investigation creation time. If not met, PolicyCondition surfaced: "Investigation lead does not meet qualification requirement per [policy rule]."
- Qualification records are included in generated CAPA reports for auditor evidence.

---

## **15. OPEN QUESTIONS**

**Resolved:**
- [x] FMEA transition path — FMISRow bridges to legacy FMEARow. Orgs choose methodology via QMS Policy. §8.7.
- [x] Synara integration contract — Investigation is a subset of the process model. Conclusions flow back. §9.6.
- [x] Control service architecture — Hybrid PolicyEvaluator (real-time) + PolicySweepEvaluator (aggregate). §4.6.
- [x] APC Frontiers integration — All 5 frontiers mapped to LOOP-001 modes and integration points. §13.
- [x] Industry-specific requirements — AS9100D, IATF 16949, FDA/13485, ISO 14971 gaps analyzed and specified. §14.

**Open (non-blocking for initial build):**
- [ ] Kaizen charter model — how do multiple investigations compose into a systemic improvement project?
- [ ] Knowledge graph entity schema — full type hierarchy for the Dynamic Process Model
- [ ] Document editor technology choice — ProseMirror/TipTap vs block-based
- [ ] Mobile experience for PCs and frontier cards
- [ ] Multi-standard health score weighting — industry-specific defaults
- [ ] Process model confidence thresholds — when is a simulation result trustworthy enough to display?
- [ ] Research module design — what data to collect from ILSSI population, IRB considerations
- [ ] CONFIG-001 standard — full configuration management specification (§14.4 deferred)
- [ ] Bayesian fault classifier forgetting factor optimization — link to EWMA lambda per F1/F5 research
- [ ] NTF analysis methodology — structured approach for No Trouble Found warranty returns (§14.7)

---

## **16. USER EXPERIENCE SPECIFICATION**

LOOP-001 defines system behavior. This section defines how that behavior surfaces to humans. Every screen described here maps directly to a mechanism, mode, or model defined above. No UX surface exists without a behavioral anchor.

### **16.1 Design Principles**

1. **Interactive instruments, not forms.** Every surface is an interactive tool that computes, visualizes, and responds — not a text field with a submit button. The reference standard is Svend Safety's frontier cards: 19 observation items across 6 categories with S/AR/U rating, severity tagging, 5S with tally vs detailed modes, operator interaction with comfort level, close-the-loop tracking. The operations calculators (OEE, Cpk, Gage R&R) are the same: interactive widgets with live computation, not data entry screens. Every LOOP-001 surface MUST meet this standard of interactivity or it does not ship.

2. **The daily surface is the accountability dashboard, not a tool.** Most QMS software opens to a document list. Svend opens to "what needs attention today." Tools are accessed FROM context (an investigation, a commitment, a signal), not from a menu.

3. **Mobile-first for verification, desktop-first for investigation.** PCs and frontier cards happen at the gemba on a phone. Investigations and document authoring happen at a desk. Don't force either to be the other.

4. **Show the loop, not the modules.** The user should see where they are in Investigate → Standardize → Verify, not which tool they're using. The loop is the navigation metaphor.

5. **Evidence over forms.** Instead of text fields labeled "root cause" and "corrective action," the system shows: hypothesis → evidence chain → conclusion. The form-shaped CAPA report is generated OUTPUT, not input.

6. **Uncertainty is visible.** When Bayesian posteriors have wide credible intervals, the UI shows that uncertainty — a faded bar, a "low confidence" label. The system never presents a number without context.

7. **Compute, don't collect.** Wherever possible, the system computes values from interactions rather than asking the user to type them. Severity posteriors update from categorical taps, not number entry. Detection rates accumulate from forced failure test pass/fail buttons, not manual scoring. The user makes judgments. The system does arithmetic.

8. **Zero manual reporting.** Engineers do not write reports. They investigate, standardize, and verify. The report engine (§5.2) assembles compliance artifacts automatically from the atoms of work already done. If an engineer is formatting a document instead of thinking about root causes, the system has failed. Every surface must have a minimum interaction density equivalent to the operations calculators — if a screen can be replaced by a spreadsheet, it's not done.

### **16.2 Accountability Dashboard (Daily Management Surface)**

**URL:** `/app/qms/` (or `/app/loop/`)
**Who uses it:** Everyone. This is the landing page for QMS users.
**Frequency:** Daily — opened at standup cadence.

**Layout: Three-column responsive grid**

```
┌──────────────────────────────────────────────────────────────┐
│  CI Readiness Score (§10)          [trend sparkline]         │
│  ███████████░░░ 72/100             ▲ +3 this week            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐│
│  │ MY COMMITMENTS  │  │ ACTIVE CONDITIONS │  │ OPEN SIGNALS││
│  │ (§3.3)          │  │ (§4.6)            │  │ (§3.1)      ││
│  │                 │  │                   │  │             ││
│  │ ⚡ Due today: 2 │  │ ⚠ 3 active       │  │ 🔴 1 crit   ││
│  │ 🔴 Overdue: 1   │  │ ℹ 5 info          │  │ 🟡 2 open   ││
│  │ ⏳ This week: 4 │  │                   │  │             ││
│  │                 │  │                   │  │             ││
│  │ [commitment     │  │ [condition list   │  │ [signal list││
│  │  list with      │  │  with severity    │  │  with source││
│  │  precondition   │  │  badges and       │  │  and triage ││
│  │  status]        │  │  "Create Signal"  │  │  status]    ││
│  │                 │  │  action]          │  │             ││
│  └─────────────────┘  └──────────────────┘  └─────────────┘│
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  ACTIVE INVESTIGATIONS                   RECENT ACTIVITY     │
│  [investigation cards with               [timeline feed:     │
│   mode indicator, owner,                  commitments         │
│   commitment count,                       fulfilled, signals  │
│   days active]                            created, docs       │
│                                           published]          │
└──────────────────────────────────────────────────────────────┘
```

**Commitment list item:**
```
┌────────────────────────────────────────────────────────┐
│ ● Revise SOP-042 Section 3              Due: Apr 3     │
│   Owner: Eric Wolters                                   │
│   Precondition: QA test data (⏳ pending — blocked by   │
│   Jane's data collection commitment, due Apr 1)         │
│   Source: Investigation INV-2026-017                     │
│   Transition: revise_document → Standardize             │
│                                     [Mark Fulfilled ▸]  │
└────────────────────────────────────────────────────────┘
```

**Key interactions:**
- Click a commitment → opens investigation workspace with that commitment highlighted
- Click "Mark Fulfilled" → system creates mode transition, opens target artifact (doc editor, FMIS, etc.)
- Click a condition → shows detail with policy rule reference + "Create Signal" action
- Click a signal → opens signal detail with "Open Investigation" or "Link to Existing" actions
- Overdue commitments show in red with duration. Blocked commitments show blocker chain.
- The precondition chain is THE daily standup artifact: "I'm blocked because X hasn't delivered Y"

### **16.3 Investigation Workspace**

**URL:** `/app/investigations/<id>/`
**Who uses it:** Quality engineers, process engineers, investigation leads
**Frequency:** Duration of investigation (days to weeks)

**Layout: Notebook-style with sidebar**

```
┌──────────┬───────────────────────────────────────────────┐
│ SIDEBAR  │  INVESTIGATION: INV-2026-017                   │
│          │  "Dimensional shift on CNC Line 3"             │
│ ▸ Graph  │  Status: ACTIVE    Lead: Eric W.               │
│ ▸ Tools  │  Mode: INVESTIGATE ████░░░░░░                  │
│ ▸ Evid.  │                                                │
│ ▸ Commit │  ┌─────────────────────────────────────────┐  │
│ ▸ History│  │ CAUSAL GRAPH (Synara)                    │  │
│ ▸ Report │  │                                          │  │
│          │  │  [interactive DAG: hypotheses as nodes,  │  │
│ ─────────│  │   evidence as edges, posterior probs     │  │
│ PROCESS  │  │   on each hypothesis, color = strength]  │  │
│ MODEL    │  │                                          │  │
│ SUBSET   │  └─────────────────────────────────────────┘  │
│          │                                                │
│ [nodes   │  ┌─────────────────────────────────────────┐  │
│  from    │  │ ENTRIES (notebook-style)                  │  │
│  process │  │                                          │  │
│  model   │  │ Mar 15 — Initial observations            │  │
│  in this │  │   [narrative + photos + data]             │  │
│  scope]  │  │                                          │  │
│          │  │ Mar 17 — Gage R&R on CMM #4              │  │
│ ─────────│  │   [linked tool output: Gage R&R result]  │  │
│ TOOLS    │  │   Evidence: MSA valid (discrimination     │  │
│          │  │   ratio = 7.2)                            │  │
│ RCA      │  │                                          │  │
│ Ishikawa │  │ Mar 19 — DOE: Temperature × Feed Rate    │  │
│ DOE      │  │   [linked tool output: factorial design]  │  │
│ SPC      │  │   Evidence: Temperature β = 0.34 ± 0.06  │  │
│ Gage R&R │  │   → Process model edge updated           │  │
│ Analysis │  │                                          │  │
│          │  │ Mar 20 — CONCLUSION                       │  │
│          │  │   Root cause: [hypothesis with P=0.92]    │  │
│          │  │   Commitments created:                    │  │
│          │  │   • Revise SOP-042 (Eric, Apr 3)          │  │
│          │  │   • Update FMIS row #47 (Jane, Mar 25)    │  │
│          │  │   • Schedule 3 PCs (Eric, Apr 10)         │  │
│          │  └─────────────────────────────────────────┘  │
└──────────┴───────────────────────────────────────────────┘
```

**Key interactions:**
- **Tool launcher (sidebar):** Click a tool → opens in-investigation context. Tool output auto-links to investigation. Evidence auto-extracted per CANON-002 weights.
- **Process model subset (sidebar):** Shows which entities from the org's process model are scoped to this investigation. Click an entity → see its current distribution, calibrated edges, and knowledge gaps. Entities with no calibrated edges are highlighted as investigation opportunities.
- **Notebook entries:** Chronological. Each entry is either narrative (written by investigator), tool output (linked from sidebar), or evidence (auto-extracted). Entries are not deletable — only supersedable per CANON-002 §6.
- **Conclusion panel:** Appears when investigation transitions to CONCLUDED. Requires: (a) a concluded hypothesis with evidence, (b) at least one Commitment created. If horizontal deployment is required per QMS Policy, surfaces "similar processes" check before conclusion.
- **Report generator (sidebar):** Click → generates CAPA report (§5) from investigation data. User selects standard template (ISO 9001, 8D, AS9100D, etc.) per QMS Policy. Report opens in document editor for review before publishing.

### **16.4 Process Confirmation (Mobile-First)**

**URL:** `/app/pc/<standard_id>/` (responsive, optimized for phone)
**Who uses it:** Supervisors, team leads, quality engineers — at the gemba
**Frequency:** Per QMS Policy schedule (e.g., weekly per standard per area)

**Layout: Vertical card stack (mobile)**

```
┌──────────────────────────────────┐
│  PC: SOP-042 "CNC Setup v3.2"   │
│  Operator: Mike R.  Area: CNC 3  │
│  Observer: Eric W.               │
│                                  │
│  ─────────────────────────────── │
│                                  │
│  Step 1: Chuck alignment         │
│  ┌────────────────────────────┐  │
│  │ Followed?  ✅ Yes  ❌ No  ⬜ NA│
│  │ Outcome?   ✅ Pass ❌ Fail ⬜ NA│
│  │ Notes: ___________________│  │
│  │ 📷 Add photo              │  │
│  └────────────────────────────┘  │
│                                  │
│  Step 2: Tool offset entry       │
│  ┌────────────────────────────┐  │
│  │ Followed?  ✅ Yes  ❌ No  ⬜ NA│
│  │ Outcome?   ✅ Pass ❌ Fail ⬜ NA│
│  │ Notes: ___________________│  │
│  │ 📷 Add photo              │  │
│  └────────────────────────────┘  │
│                                  │
│  ... (remaining steps)           │
│                                  │
│  ─────────────────────────────── │
│  Observer Notes:                 │
│  _______________________________│
│                                  │
│  Improvements Observed:          │
│  _______________________________│
│                                  │
│  ─────────────────────────────── │
│  DIAGNOSTIC SUMMARY              │
│  Steps followed: 8/10            │
│  Outcomes correct: 9/10          │
│  Diagnosis: STANDARD UNCLEAR     │
│  (Step 4, 7 not followed —       │
│   outcome still correct both     │
│   times → standard may be        │
│   impractical for these steps)   │
│                                  │
│              [Submit PC ▸]       │
└──────────────────────────────────┘
```

**Interaction model (reference: Frontier Card depth):**

- **Steps auto-populated** from the linked ControlledDocument's key steps. Each step shows the step text, key point, and reason-why from the JIB. The observer reads the standard and watches.
- **Single-tap rating** — large touch targets. Followed? (Yes/No/NA). Outcome? (Pass/Fail/NA). Two taps per step. No typing required for the core observation.
- **Deviation classification** — when "No" or "Fail" is tapped, a severity selector slides in (same pattern as frontier card S/AR/U with C/H/M/L severity). Single tap. The system knows "not followed + moderate" vs "not followed + critical."
- **Photo capture** — device camera, auto-linked to the specific step. Annotate inline (arrow, circle, text callout — same tools as document editor). Photo is evidence, not decoration.
- **Operator interaction** — after observation, structured interaction prompts (adapted from STOP methodology): "What did you observe going well?", "What challenges do you face with this step?", "Is there anything about this standard you'd change?" Operator responses are captured as structured data, not free text.
- **Comfort level** — operator comfort with being observed: comfortable / neutral / uncomfortable (same model as frontier card). Tracks observer effectiveness.
- **Diagnostic summary** — auto-computed LIVE as the observer taps. The matrix (§7.1: followed × outcome) runs in real time. "Standard unclear" diagnosis appears as soon as the pattern emerges — the observer sees the diagnosis forming, not after submission.
- **Trend overlay** — if this operator has prior PCs on this standard, the trend is visible: "Pass rate: 4/5 → now 3/5 on this standard." Real-time context.
- **Submit → PolicyEvaluator** checks thresholds (§4.6) immediately. If threshold breach → condition surfaced on accountability dashboard. Observer sees: "This submission triggered a policy condition: [pc.retraining_threshold]."
- **Close-the-loop** — same pattern as frontier card close-the-loop: was feedback given? (immediate / within 24h / pending / not done). Tracked and visible.

**What this is NOT:** A form with text fields. The observer taps, photographs, and talks. The system computes, diagnoses, and surfaces conditions. The entire PC should take 5-10 minutes on a phone at the gemba, not 30 minutes at a desk filling out a form.

### **16.5 FMIS View (Bayesian FMEA)**

**URL:** `/app/fmis/<id>/` (or within investigation workspace)
**Who uses it:** Process engineers, quality engineers
**Frequency:** During investigations, periodic reviews

**Layout: Table with expandable rows + posterior visualizations**

```
┌─────────────────────────────────────────────────────────────────┐
│ FMIS: CNC Line 3 Process                     [Methodology: ▼] │
│ Investigation: INV-2026-017                    svend_bayesian   │
│                                                                 │
│ FM │ Effect │ Cause │ Sev │ Occ │ Det │ RPN │ Evid │ Status    │
│────┼────────┼───────┼─────┼─────┼─────┼─────┼──────┼─────────│
│ ▸ Tool     │ Dim.  │ Wear│ 4.2 │ 2.8 │ 3.1 │  37 │ 14   │ Active │
│   wear     │ shift │     │ ██░░│ █░░░│ ██░░│     │      │        │
│ ▸ Material │ Surf. │ Lot │ 2.1 │ 5.3 │ 1.4 │  16 │  3   │ Active │
│   hardness │ rough │ var │ █░░░│ ███░│ █░░░│     │      │        │
│ ▸ Coolant  │ Ther. │ Flow│ 6.8 │ 1.2 │ ??  │  —  │  0   │ Gap    │
│   failure  │ damage│ loss│ ████│ █░░░│ ░░░░│     │      │        │
└─────────────────────────────────────────────────────────────────┘
```

**Expand a row:**
```
┌─────────────────────────────────────────────────────────────────┐
│ ▾ Tool wear → Dimensional shift                                 │
│                                                                 │
│ SEVERITY (Categorical-Dirichlet)        OPERATIONAL DEFINITIONS │
│ ┌────────────────────────────────┐      Failure mode:           │
│ │ Negligible ██░░░░░░░░ 12%     │        → Entity: "Tool wear  │
│ │ Minor      ████████░░ 48%     │           rate" [linked ▸]    │
│ │ Moderate   ██████░░░░ 30%     │      Effect:                  │
│ │ Severe     ██░░░░░░░░  8%     │        → Entity: "Part OD     │
│ │ Catastroph ░░░░░░░░░░  2%     │           dimension" [linked] │
│ │ E[Sev] = 2.4  (14 observations)│     Cause:                   │
│ └────────────────────────────────┘       → Entity: "Insert      │
│                                             life" [linked ▸]    │
│ OCCURRENCE (Beta-Binomial)                                      │
│ ┌────────────────────────────────┐     CONTROLS                 │
│ │ Rate: 2.3% [1.1%, 4.2%]       │     Prevention: Tool life     │
│ │ 12 failures / 520 units        │       counter with preset     │
│ │ ████████░░░░░░░░░░░░ 2.3%     │     Detection: CMM 100%       │
│ └────────────────────────────────┘       inspection post-op     │
│                                                                 │
│ DETECTION (Beta-Binomial)              FORCED FAILURE TESTS     │
│ ┌────────────────────────────────┐     2026-03-10: 6/8 detected │
│ │ Rate: 75% [55%, 89%]          │     2026-02-15: 5/6 detected │
│ │ 11 detected / 14 injected      │     → Next test scheduled:   │
│ │ ████████████████░░░░ 75%       │       2026-04-01             │
│ └────────────────────────────────┘                              │
│                                                                 │
│ [View in Process Model ▸]  [Schedule Forced Failure ▸]          │
│ [Add Consequence Observation ▸]                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key interactions:**
- **`??` detection (no data):** Row shows "Gap" status. System surfaces: "No forced failure test data — detection score is unknown." Not a default, not a guess — an explicit gap.
- **Entity links:** Click an operational definition → navigates to process model entity with its distribution, calibrated edges, measurement system.
- **Posterior bars:** Horizontal bars show the distribution. Wider = more uncertain. Narrow = more data. The NUMBER is the posterior mean. The BAR is the credible interval. Users learn to trust narrow bars and question wide ones.
- **"Add Consequence Observation":** Opens a quick-entry form: classify this event into severity category (1-5) + link to investigation/NCR/complaint. One click adds evidence to the Dirichlet posterior.
- **Methodology selector:** Dropdown switches between `aiag_4th` (manual integers), `svend_bayesian` (posteriors with integer mapping), `svend_full` (posteriors + operational definitions required). Per QMS Policy.

### **16.6 Document Editor**

**URL:** `/app/iso-docs/<id>/edit/`
**Who uses it:** Process engineers, document authors, quality managers
**Frequency:** When creating or revising standards (from investigation commitments)

**Layout: Split-pane — editor left, preview right**

```
┌────────────────────────────┬────────────────────────────┐
│ EDITOR                     │ LIVE PREVIEW               │
│                            │                            │
│ ┌────────────────────────┐ │ ┌────────────────────────┐│
│ │ ≡ 1. Purpose           │ │ │ SOP-042               ││
│ │   [paragraph content]  │ │ │ CNC Setup Procedure    ││
│ │                        │ │ │ Rev 3.2 | Apr 2026     ││
│ └────────────────────────┘ │ │                        ││
│ ┌────────────────────────┐ │ │ 1. Purpose             ││
│ │ ≡ 2. Scope             │ │ │ This procedure...      ││
│ │   [paragraph content]  │ │ │                        ││
│ └────────────────────────┘ │ │ 2. Scope               ││
│ ┌────────────────────────┐ │ │ Applies to...          ││
│ │ ≡ 3. Important Steps   │ │ │                        ││
│ │   ┌──────────────────┐ │ │ │ 3. Important Steps     ││
│ │   │ Step 3.1: Chuck  │ │ │ │                        ││
│ │   │ alignment         │ │ │ │ 3.1 Chuck alignment   ││
│ │   │ Key Point: Center │ │ │ │ ┌──────────────────┐  ││
│ │   │ within 0.002"     │ │ │ │ │ [photo: chuck    │  ││
│ │   │ Reason: Prevents  │ │ │ │ │  alignment with  │  ││
│ │   │ runout > spec     │ │ │ │ │  indicator]      │  ││
│ │   │ [📷 photo]        │ │ │ │ └──────────────────┘  ││
│ │   │ [🔗 linked to     │ │ │ │ Key Point: Center     ││
│ │   │  Entity: "Chuck   │ │ │ │ within 0.002"         ││
│ │   │  runout"]         │ │ │ │ Why: Prevents runout  ││
│ │   └──────────────────┘ │ │ │ > specification        ││
│ │   [+ Add Step]         │ │ │                        ││
│ └────────────────────────┘ │ └────────────────────────┘│
│                            │                            │
│ [+ Add Section ▼]         │ [Export PDF] [Export Word]  │
│  Heading | Paragraph |     │                            │
│  JIB Step | Checklist |    │                            │
│  Image | Table |           │                            │
│  Signature Block           │                            │
│                            │                            │
│ ──────────────────────     │                            │
│ SOURCE INVESTIGATION       │                            │
│ INV-2026-017 [view ▸]     │                            │
│ "Dimensional shift on      │                            │
│  CNC Line 3"               │                            │
│ This revision addresses    │                            │
│ root cause: tool wear      │                            │
│ detection gap.             │                            │
└────────────────────────────┴────────────────────────────┘
```

**Key interactions:**
- **Drag sections** (≡ handle) to reorder. Sections are the unit of authoring — not free-form rich text.
- **JIB Step template:** Important Steps → Key Points → Reasons Why. TWI format baked in. Each step can have photos with inline annotation (arrows, callouts, highlights).
- **Photo drop-in:** Drag photo from file system or paste from clipboard. Photo renders inline with the step. Annotation tools overlay on the photo (arrows, boxes, text callouts).
- **Entity links:** Steps can link to knowledge graph entities. "Center within 0.002" links to the entity "Chuck runout" with its operational definition, measurement method, and linked equipment. These are the same entities FMIS references.
- **Investigation link:** When the document revision was created from an investigation commitment, the source investigation is shown. Auditors can trace: why did this SOP change → investigation → root cause → evidence.
- **Publish flow:** Save → Submit for Review → Approve (e-signature) → Publish → auto-creates TrainingRequirement (§3.2 mode transition: Standardize → Verify).
- **AI drafting (future):** "Draft from investigation" button → Anthropic generates section content from investigation entries, evidence, and conclusion. Author edits. Not auto-published.

### **16.7 QMS Policy Configuration**

**URL:** `/app/qms/settings/policies/` (admin surface)
**Who uses it:** Quality managers, system administrators
**Frequency:** Setup and periodic review

**Layout: Cloudflare-style rule list with detail panel**

```
┌─────────────────────────────────────────────────────────────┐
│ QMS POLICIES                                  [+ Add Rule]   │
│                                                              │
│ ┌──────────────────────────────────────────────────────────┐│
│ │ Scope              │ Rule Key              │ Status       ││
│ ├────────────────────┼───────────────────────┼─────────────┤│
│ │ Process Confirm.   │ pc.retraining_thresh  │ ● Active     ││
│ │ Process Confirm.   │ pc.escalation         │ ● Active     ││
│ │ FMIS               │ fmis.methodology      │ ● Active     ││
│ │ FMIS               │ fmis.review_frequency │ ● Active     ││
│ │ Investigation      │ inv.horizontal_deploy │ ● Active     ││
│ │ Investigation      │ inv.lead_qualification│ ○ Inactive   ││
│ │ Training           │ trn.reflection_thresh │ ● Active     ││
│ │ Calibration        │ cal.overdue_threshold │ ● Active     ││
│ │ CAPA Report        │ capa.report_standard  │ ● Active     ││
│ │ CAPA Report        │ capa.include_trending │ ● Active     ││
│ └──────────────────────────────────────────────────────────┘│
│                                                              │
│ RULE DETAIL: pc.retraining_threshold                         │
│ ┌──────────────────────────────────────────────────────────┐│
│ │ Scope: process_confirmation                               ││
│ │ Parameters:                                               ││
│ │   pass_rate_threshold:    [0.80 ▼]                        ││
│ │   trailing_window:        [5    ▼] PCs                    ││
│ │   cooldown_days:          [30   ▼]                        ││
│ │   escalate_after_n:       [3    ▼] operators              ││
│ │   escalation_window_days: [60   ▼]                        ││
│ │                                                           ││
│ │ Linked standard: ISO 9001 §7.2                            ││
│ │ Effective date: 2026-04-01                                ││
│ │ Approved by: Eric Wolters                                 ││
│ │ Version: 1                                                ││
│ │                                                           ││
│ │ [Save as Draft]  [Publish (creates ControlledDocument)]   ││
│ └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

**Key interactions:**
- Each rule has typed parameter fields (numbers, dropdowns, booleans) — not free text
- "Publish" saves the rule AND generates a ControlledDocument version (the policy artifact for auditors)
- Inactive rules are saved but not evaluated by PolicyEvaluator
- Version history: click any rule → see all prior versions with effective dates and who approved
- Conflict detection: if two rules contradict, the save button shows a warning with explanation

### **16.8 Signal Triage**

**URL:** `/app/qms/signals/` (or from accountability dashboard)
**Who uses it:** Quality managers, investigation leads
**Frequency:** As signals are created (daily to weekly)

**Layout: Kanban-style columns**

```
┌───────────────┬───────────────┬───────────────┬──────────────┐
│ UNTRIAGED (3) │ ACKNOWLEDGED  │ INVESTIGATING │ RESOLVED (12)│
│               │ (2)           │ (4)           │              │
│ ┌───────────┐ │ ┌───────────┐ │ ┌───────────┐ │              │
│ │🔴 SPC     │ │ │🟡 PC      │ │ │ INV-017   │ │              │
│ │violation  │ │ │threshold  │ │ │ linked ▸  │ │              │
│ │CNC Line 3 │ │ │SOP-042    │ │ │           │ │              │
│ │2 hrs ago  │ │ │3 operators│ │ │           │ │              │
│ │           │ │ │failing    │ │ │           │ │              │
│ │[Ack] [Inv]│ │ │           │ │ │           │ │              │
│ │[Dismiss]  │ │ │[Open Inv] │ │ │           │ │              │
│ │[Link]     │ │ │[Link]     │ │ │           │ │              │
│ └───────────┘ │ └───────────┘ │ └───────────┘ │              │
│ ...           │               │ ...           │              │
└───────────────┴───────────────┴───────────────┴──────────────┘
```

**Key interactions:**
- **Acknowledge:** "I've seen this, will triage later." Moves to acknowledged column. Prevents duplicate signals from the same condition.
- **Open Investigation:** Creates new investigation pre-populated with signal source data. Signal status → investigating.
- **Link to Existing:** Select an active investigation. Signal linked via UUID. Status → investigating.
- **Dismiss:** Requires reason (text field). Dismissals are auditable. High dismiss rates surface in bias detection (§12).
- Drag-and-drop between columns (desktop). Tap-to-action (mobile).

### **16.9 Auditor Portal**

**URL:** `/audit/<token>/` (no authentication required — time-limited token)
**Who uses it:** External auditors during certification/surveillance audits
**Frequency:** 1-2 times per year per auditor

**Layout: Clean read-only view organized by standard clause**

```
┌─────────────────────────────────────────────────────────────┐
│ SVEND — Auditor Portal            [Standard: ISO 9001 ▼]    │
│ Organization: Acme Manufacturing   Expires: 2026-04-15      │
│                                                              │
│ CI READINESS: 78/100 ████████░░ (▲ +12 over 6 months)      │
│                                                              │
│ ┌──────────────────────────────────────────────────────────┐│
│ │ Clause 10.2 — Nonconformity and Corrective Action        ││
│ │                                                          ││
│ │ Investigations: 14 (12 concluded, 2 active)              ││
│ │ Avg time to conclusion: 8.3 days                         ││
│ │ % with statistical evidence: 71%                         ││
│ │ Recurrence rate: 8% (↓ from 15%)                         ││
│ │                                                          ││
│ │ ▸ INV-2026-017: Dimensional shift CNC Line 3             ││
│ │   Signal → Investigation → Fix → SOP revision →          ││
│ │   Training (6/6 complete, reflections captured) →         ││
│ │   3 PCs completed (all pass)                             ││
│ │   [View full chain ▸]  [Generated CAPA report ▸]        ││
│ │                                                          ││
│ │ ▸ INV-2026-012: Supplier material variability            ││
│ │   ...                                                    ││
│ └──────────────────────────────────────────────────────────┘│
│                                                              │
│ ┌──────────────────────────────────────────────────────────┐│
│ │ Clause 7.2 — Competence                                  ││
│ │                                                          ││
│ │ Training coverage: 94% current                           ││
│ │ Reflections captured: 89% of completions                 ││
│ │ Avg competency level: 2.8 / 4.0                         ││
│ │ [View training matrix ▸]                                 ││
│ └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

**Key interactions:**
- Standard selector: switch between ISO 9001, IATF 16949, AS9100D, ISO 13485 — same data reorganized by clause
- Click "View full chain": expands the signal → investigation → standardize → verify chain with all linked artifacts
- Click "Generated CAPA report": opens the auto-assembled report for that investigation
- No edit capability anywhere. Pure evidence presentation.
- Token expires after org-configured period. Access logged in audit trail.

---

## **REVISION HISTORY**

| Version | Date | Author | Changes |
|---|---|---|---|
| 0.1 | 2026-03-27 | Eric + Claude | Initial design. Three mechanisms formalized. FMIS schema specified. Bayesian S/O/D (Beta-Binomial detection/occurrence, Categorical-Dirichlet severity). QMS Policy with hybrid control service. Dynamic Process Model with investigation-as-subset. 5 APC Frontiers integrated (§13). Industry requirements for AS9100D, IATF 16949, FDA/13485 (§14). 5 open questions resolved, 10 remaining. |
