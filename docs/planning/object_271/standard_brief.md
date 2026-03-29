# OLR-001: Organizational Learning Rate — Standard Brief

**Date:** 2026-03-29
**For:** Claude Cowork (drafting session)
**Lineage:** CCS 1948 → Shingo/Ohno → Protzman → Eric Wolters → SVEND → OLR-001

## Critical Constraint: Platform Independence

**This is a STANDARD, not a product manual.** Every requirement must be auditable by a third party using only observation, interviews, and record inspection. An organization could satisfy OLR-001 using spreadsheets and whiteboards — they'd be slower and worse at it, but they could do it.

The standard never references software features, screen layouts, API endpoints, or specific tools. SVEND is the most efficient implementation. It is not the only one.

**Two-layer architecture:**
- **Layer 1: Requirements** — what the organization must DO. Platform-agnostic. Auditable with a clipboard.
- **Layer 2: Technical Annexes** — how to do it with maximum rigor. Bayesian math, entropy formulas, graph schemas. Optional for certification. Required for Level 4 maturity.

This mirrors ISO's pattern: the standard is the requirements, the guidance documents are the how.

## What This Is

A quality system standard built from first principles around organizational learning — not compliance forms. It measures whether an organization is learning about its processes, how fast, and whether that learning is being encoded, verified, and maintained.

It is NOT a rebrand of ISO 9001. ISO 9001 is a compatibility mapping TO this standard, not the other way around.

## Core Thesis

Quality management is not form completion. It is the rate at which an organization converts ignorance about its processes into calibrated, evidence-based knowledge — and maintains that knowledge against entropy.

The standard defines:
- **What to know** — structured process knowledge with causal relationships and evidence provenance
- **How to learn** — the closed loop (detect → investigate → standardize → verify)
- **How to measure learning** — the proportion of process knowledge that is evidence-based and current
- **How to prove it** — evidence provenance and audit protocol

## The Three Concerns

```
Process Knowledge — what the organization knows (structured, evidence-based, maintained)
Learning System   — how the organization learns (detect → investigate → standardize → verify)
Compliance        — what the organization demonstrates (audit evidence, regulatory mappings)
```

These are peer systems. Compliance does not own Process Knowledge or the Learning System. It reads from them.

## What Contemporary QMS Components Become

The standard rejects several contemporary QMS concepts in favor of first-principles alternatives:

### CAPA → Investigation + Generated Report

There is no CAPA process. There is an investigation that produces knowledge. The investigation:
1. Scopes a subgraph of the process model
2. Discovers new causal relationships (new nodes, new edges)
3. Calibrates existing edges (DOE, analysis, observation)
4. Writes findings back to the graph

The compliance artifact called "CAPA" is auto-assembled from investigation data for auditors who need to see it in that format. It is a VIEW on the investigation, not a separate process.

**Standard requirement:** "The organization SHALL investigate signals using a structured methodology that produces calibrated evidence on the process knowledge graph. A compliance report in CAPA format SHALL be generatable from any concluded investigation."

### NCR → Signal

A nonconformance record is one type of Signal — an event demanding attention. The standard unifies all signal sources:

| Signal source | Traditional name | In OLR-001 |
|---------------|-----------------|------------|
| Process defect | NCR | Signal (source: process) |
| Customer feedback | Complaint | Signal (source: customer) |
| SPC alarm | Out-of-control | Signal (source: monitoring) |
| Audit finding | Finding | Signal (source: verification) |
| Graph staleness | — | Signal (source: knowledge_decay) |
| Graph contradiction | — | Signal (source: evidence_conflict) |
| Supplier issue | Supplier NCR | Signal (source: supply_chain) |
| Operator observation | — | Signal (source: gemba) |

The Loop handles all of them the same way. The distinction between signal types matters for routing and severity, not for process.

**Standard requirement:** "The organization SHALL maintain a unified signal detection system that captures events from all sources requiring investigation. Signal classification SHALL determine routing and urgency, not process."

### Management Review → Continuous Knowledge Dashboard

The quarterly management review is replaced by continuous visibility into process knowledge state. The "review" becomes a decision point, not an information-gathering exercise.

The graph health metrics ARE the management review inputs:
- Entropy trend (is the organization learning?)
- Calibration rate (what % of claimed relationships have evidence?)
- Staleness distribution (how current is the knowledge?)
- Contradiction count (where does the model disagree with reality?)
- Signal resolution velocity (how fast does the Loop turn?)
- Investigation writeback rate (are investigations producing knowledge?)

**Standard requirement:** "Leadership SHALL have continuous access to process knowledge metrics. Periodic review meetings SHALL focus on decisions and resource allocation based on these metrics, not on data gathering."

### Document Control → Standardization Artifacts

Controlled documents are outputs of the Standardize mode. When the Loop learns something, it encodes that knowledge as a standard (work instruction, procedure, specification). The document is the artifact. The Loop is the process that produces it.

**Standard requirement:** "The organization SHALL encode investigation conclusions as controlled standardization artifacts. These artifacts SHALL be linked to the process knowledge graph elements they govern."

### Training → Knowledge Gap Response

Training requirements are derived from graph gaps, not from static job descriptions. When the graph shows uncalibrated edges in an area where an employee works, that's a training signal.

**Standard requirement:** "Competency requirements SHALL be informed by process knowledge gaps in the employee's area of responsibility. Training effectiveness SHALL be assessed against graph calibration improvement in those areas."

### FMEA → Graph View

FMEA is not a standalone tool. It is a filtered view of the process knowledge graph showing failure mode nodes with their upstream causes and downstream effects. S/O/D scores are derived from edge posteriors.

**Standard requirement:** "Risk assessment SHALL be maintained as a continuous property of the process knowledge graph, not as a periodic document. Severity, occurrence, and detection SHALL be evidence-based posteriors, with manual assessment permitted only where empirical data is unavailable."

## What Is New (No ISO Equivalent)

### Structured Process Knowledge

No existing standard requires this. The organization must maintain:
- **A structured representation of process relationships** — what affects what, with direction. This could be a directed graph in software, a wall chart, a structured spreadsheet, or a database. The form doesn't matter. The structure does.
- **Evidence provenance on every claimed relationship** — for every "X causes Y," you can show the evidence: what study, what data, when, by whom. If there's no evidence, the relationship is an assertion (hypothesis), not knowledge.
- **Quantified confidence** — how confident are you in each relationship? At minimum: high/medium/low with last validated date and evidence source. With maximum rigor: Bayesian posterior probability updated from evidence (see Annex C).
- **Visible gaps** — uncalibrated assertions, stale relationships, contradictions. The organization must be able to answer "what don't we know?" at any time.
- **Growth from problems** — you don't map everything upfront. Knowledge grows backwards from the problems you investigate.

**Standard requirement:** "The organization SHALL maintain structured process knowledge in which every claimed causal relationship has traceable evidence provenance and quantified confidence. Relationships without empirical evidence SHALL be identified as knowledge gaps."

### Process Knowledge Health

The proportion of process knowledge that is uncalibrated, stale, or contradicted is measurable. An organization that is learning has improving knowledge health — more relationships are calibrated, fewer are stale, contradictions are investigated.

This is countable by hand if the process knowledge structure is on a whiteboard. Count the relationships. Count those with evidence. That's your calibration rate. Count those where the evidence is older than your review cycle. That's your staleness rate. The formal entropy computation goes in Annex C for organizations that want mathematical rigor.

**Standard requirement:** "The organization SHALL track the proportion of its process knowledge that is evidence-based and current. Sustained decline in knowledge health SHALL trigger management action."

### Maturity Levels

Defined by what the auditor finds at inspection, not by what a dashboard shows:

| Level | Name | What the auditor sees |
|-------|------|----------------------|
| 1 | **Structured** | Process knowledge is documented in structured form. Relationships are identified. Most lack empirical evidence. The organization knows what it doesn't know. |
| 2 | **Learning** | Evidence is accumulating. Some relationships have been calibrated since last audit. Investigations are producing new knowledge. Knowledge health is improving. |
| 3 | **Sustaining** | Few stale relationships. Contradictions investigated within threshold. Knowledge health is stable or improving over sustained period. The learning system is self-maintaining. |
| 4 | **Predictive** | The organization demonstrates predictive capability — "we predicted X from our process model, measured Y, they agree within Z" — for critical relationships. Requires quantitative process model (Annex C rigor). |

Levels 1-3 are auditable with a clipboard. Level 4 requires computational rigor (the Technical Annexes).

An organization at Level 3 that stops investigating drops to Level 2 — knowledge decays without maintenance. Certification is not permanent — it reflects the current state of organizational learning.

**Standard requirement:** "Certification level SHALL be determined by the current state and trajectory of process knowledge health, not by point-in-time document review."

### Forced Failure Testing

No existing standard requires you to intentionally inject failures to test your detection systems. OLR-001 does.

**Standard requirement:** "For critical detection controls, the organization SHALL conduct forced failure tests at a frequency determined by risk. Detection posteriors SHALL be updated from test results."

### Process Confirmation (David Mann)

Go to the floor. Watch the process. Is the standard being followed? Is the outcome correct? This is the Verify mode.

**Standard requirement:** "The organization SHALL verify that standardized processes produce expected outcomes through direct observation. Process confirmation findings SHALL update the process knowledge graph as evidence."

### Pre-Production Knowledge Design (3P + QFD)

Before production begins, the organization builds the knowledge structure AND calibrates it. This is 3P (Production Preparation Process) expanded to include knowledge preparation, not just process preparation.

**The sequence:**

1. **Customer requirements → graph structure (QFD).** House of Quality translates voice of customer into technical characteristics. Each strong relationship in the QFD matrix becomes an FMIS row — a claimed relationship between a customer requirement (specification node) and a process parameter. Customer classification (critical/major/minor) flows down to node tiers automatically.

2. **Process design → graph expansion (3P).** Design the process to meet requirements. Each process step, each parameter, each potential failure mode becomes a node. Relationships between steps become edges. The graph grows from the design, seeded with engineering assertions.

3. **Physical validation → graph calibration (moonshining).** Build it. Break it. Understand it. Every moonshining cycle is an investigation that produces evidence. Run the process at different settings, measure the outcomes, record the results on evidence forms. Update the FMIS rows from assertion to calibrated. This is the Loop running BEFORE production starts.

4. **Special process qualification → high-strength evidence.** For heat treat, chemical processes, composite cure, welding — formal qualification testing produces DOE-level evidence on critical edges. The qualification report IS the evidence form. References FMIS row numbers.

5. **Configuration boundaries defined.** Which product variants affect which graph nodes? A table mapping configurations to process settings. When engineering issues a change order, the affected nodes are identified and their edges flagged for recalibration. This handles configuration management through the graph, not through a separate system.

6. **First article verification → full chain validation.** First article inspection walks the entire graph from input to output. Verifies each relationship in the chain. Results calibrate any remaining edges. FAI report references FMIS rows.

7. **Control plan derived from calibrated graph.** Each control plan item references the FMIS row it monitors. The control plan IS a view of the graph filtered to "things we need to monitor in production." It's not a separate document created in parallel — it's generated from the knowledge structure.

**After step 7:** Production begins with a partially-calibrated graph, a control plan derived from it, and clear gaps identified for ongoing calibration. The organization enters production at Level 2. The Loop continues to calibrate during production.

**Standard requirement:** "For new products or processes, the organization SHALL design the process knowledge structure before production begins. Customer requirements SHALL be translated into classified knowledge elements. Process relationships SHALL be validated through physical builds or qualification testing. The production control plan SHALL be derived from the validated knowledge structure."

### Node Classification Tiers

Not all process knowledge is equal. A relationship affecting flight safety requires more rigor than one affecting paint color. The standard defines classification tiers that determine evidence minimums and maintenance frequency:

| Tier | Scope | Evidence minimum | Staleness threshold | Cannot be assertion-only after |
|------|-------|-----------------|-------------------|-------------------------------|
| **Critical** | Safety, key characteristics, regulatory | Controlled experiment or formal qualification | 90 days | Level 1 |
| **Major** | Performance, fit, function | Structured observation with data | 180 days | Level 2 |
| **Minor** | Cosmetic, non-functional, convenience | Any externalized evidence | 365 days | Level 3 |

Classification is set during Pre-Production Knowledge Design (from QFD customer requirement classification) and can be elevated by investigation findings (a relationship thought minor turns out to be safety-relevant).

The configuration service sets the specific thresholds per industry:
- Aerospace preset: critical = 90 days, major = 120 days, minor = 365 days
- Automotive preset: critical = 180 days, major = 365 days, minor = 730 days
- General manufacturing: critical = 180 days, major = 365 days, minor = no limit

**Standard requirement:** "The organization SHALL classify all process knowledge elements by consequence tier. Evidence requirements and maintenance frequency SHALL be proportional to classification. Critical elements SHALL NOT remain assertion-only beyond Level 1 maturity."

### Customer Satisfaction as Process Knowledge Health

Customer satisfaction is not measured by surveys. It is measured by the health of the process knowledge elements that affect what customers receive.

**The model:**

1. **Customer-facing nodes identified** — during QFD, customer requirements map to specification nodes. These are the quality characteristics customers care about.

2. **Those nodes monitored** — SPC, process confirmation, forced failure testing on the edges feeding customer-facing nodes.

3. **Proactive/reactive ratio** — the primary satisfaction metric. Of all quality events affecting customer-facing nodes: what percentage were detected by the organization's monitoring BEFORE the customer reported them? Computed from signal source timestamps.
   - 90%+ proactive → organization catches problems before customers see them
   - 50/50 → organization and customers detect at similar rates
   - <50% proactive → customers are finding your problems for you

4. **Customer feedback as signal source** — complaints, returns, field reports enter the signal system. Each maps to customer-facing nodes. Investigation resolves the signal and updates the graph.

5. **Signal resolution velocity on customer-facing nodes** — how fast does the organization respond when something affecting customers changes?

This model is computable from operational data. It's auditable — the auditor samples customer complaints, checks if there was a preceding internal signal, checks the timestamps. It replaces subjective satisfaction surveys with objective process performance measurement.

**Standard requirement:** "The organization SHALL identify which process knowledge elements directly affect product or service quality received by customers. Satisfaction SHALL be assessed by the ratio of proactive internal detection to reactive customer reporting on those elements. The organization SHALL track signal resolution velocity on customer-facing elements."

**Scope note:** OLR-001 addresses product and service quality satisfaction. Commercial satisfaction (pricing, terms, delivery lead times) is outside the scope of this standard.

### The Cultural Inversion — OLR-001's Secret Weapon

ISO 9001 is a compliance burden. Nobody loves it. OLR-001 inverts this.

**In ISO:** Knowledge flows down. Engineers write procedures. Operators follow them. The quality system is something imposed on workers by management.

**In OLR-001:** Knowledge flows in ALL directions. Operators contribute assertions from 20 years of experience. Engineers calibrate those assertions with DOE. Leadership allocates resources based on gaps. Everyone contributes to the knowledge structure. Everyone's contribution is traceable.

The machinist who says "humidity causes short shots after lunch" — in ISO, that's tribal knowledge. In OLR-001, that's a signal. It enters the FMIS as an assertion with her name and date. If an investigation confirms it, the control plan now includes a humidity check, and the operator can see the chain: her assertion → the DOE that validated it → the standardized practice that resulted. Her experience was taken seriously, subjected to rigor, and either validated or refined. Not dismissed.

**This reframes every interaction:**

- **Gemba walks** become knowledge harvesting. The leader walks the floor asking "what do you know that we haven't captured?" Every walk should produce FMIS rows — assertions to be investigated, not corrective actions to be imposed. The walk IS a signal source.

- **Training** becomes bidirectional. The new engineer learns from the graph what the organization knows. The experienced operator teaches the graph what she knows. Both are learning. Both are contributing.

- **Communication** isn't a managed process — it's what happens naturally when someone's assertion enters the system, gets investigated, and flows back as a standard they can trace to their own observation. That IS communication. That IS respect for people. That IS Deming's System of Profound Knowledge in practice.

- **Leadership** isn't reviewing reports — it's creating and sustaining the culture and systems that make knowledge contribution safe, valued, and continuous. Configuration-as-policy means leadership has committed to how the system works. The RACI means everyone knows their role. The dashboard means leadership sees whether the system is learning.

**This is the adoption advantage.** ISO is something you endure. OLR-001 is something you USE. The paper forms aren't compliance artifacts — they're how you externalize what you know so it survives you. The standard doesn't demand more work — it gives existing work a structure that compounds.

**Standard requirement:** "The organization SHALL maintain a culture in which process knowledge contribution is expected from all personnel. Assertions from any source SHALL enter the knowledge structure with attribution. Gemba interactions, operator observations, and practitioner experience SHALL be treated as signal sources for investigation. The organization SHALL demonstrate that contributed assertions are investigated, not dismissed."

### Continuous Knowledge Capture

Knowledge doesn't get captured when someone retires. It gets captured continuously, from everyone, all the time.

**Sources of continuous knowledge input:**
- Gemba walks (leader observations → FMIS assertions)
- Operator daily experience (process anomalies → signals)
- Process confirmation (standard followed + outcome verified → evidence)
- Maintenance events (equipment changes → staleness triggers on downstream edges)
- Material lot changes (new supplier or lot → staleness triggers on material nodes)
- Engineering changes (design revision → affected nodes flagged for recalibration)
- Seasonal/environmental shifts (observed patterns → environmental factor assertions)

**Knowledge loss prevention:** If an experienced person's knowledge is in the FMIS with their name on it, their retirement doesn't erase it. The assertions persist. The evidence persists. Their successor inherits a structured map of what the predecessor knew, with clear gaps showing what was experience-only vs what was calibrated.

**Standard requirement:** "The organization SHALL capture process knowledge continuously, not periodically. Personnel transitions SHALL NOT result in loss of process knowledge. The knowledge structure SHALL be the persistent repository for practitioner expertise, with attribution and evidence provenance on all entries."

### Staleness Triggers

Evidence is stale when the conditions under which it was generated no longer represent the current process state. Time-based thresholds are the safety net for changes you didn't notice. Explicit triggers are the primary mechanism.

**Explicit staleness triggers:**
- New material lot or supplier change → edges downstream of material nodes
- Equipment maintenance, replacement, or recalibration → edges involving that equipment
- Personnel change on skill-dependent operations → edges where operator is a factor
- Environmental shift (seasonal, facility) → edges involving environmental factors
- Engineering change order → edges on affected process parameters
- Process parameter change (new setpoints) → edges from that parameter

When a trigger fires, affected edges are flagged as POTENTIALLY stale. Not automatically invalid — potentially. Someone must look and decide whether the change materially affects the relationship. The investigation may conclude "no impact" (edge stays calibrated) or "recalibration needed" (evidence is stale, investigate).

**Standard requirement:** "The organization SHALL identify process changes that may invalidate existing evidence. Changes to inputs, equipment, methods, personnel, or environment SHALL trigger review of affected process knowledge elements. Time-based review cycles SHALL serve as secondary staleness detection for changes not captured by explicit triggers."

### Multi-Site Knowledge Transfer

Assertions transfer across sites. Evidence does not. Calibration does not.

If Site A discovers "temperature affects viscosity with effect size 0.3," Site B can adopt the ASSERTION: "temperature probably affects viscosity at our site too." That's a research direction, not a calibration. Site B must generate its own evidence. Even if processes are nominally identical, the assumption of horizontal transferability is unwarranted — equipment differs, materials differ, environment differs.

**Exception: measurement system evidence.** If identical equipment (same model, same master, same calibration standard) has been validated via Gage R&R at Site A, that evidence MAY transfer to Site B as a prior — with mandatory local verification. Measurement capability is equipment-specific, not process-specific. The assertion transfers as: "validated on identical equipment at Site A, requires local verification of installation environment."

**On merger/acquisition:** The graph persists at the organizational unit that owns the process. If two sites merge, you have two graphs for what is now one site. You investigate where they agree and disagree. Disagreement reveals that two nominally identical processes actually differ. That's knowledge the merger surfaced. If a product line is divested, the graph goes with it — fully provenanced, so the acquirer can trace every assertion to its source and decide what to trust.

**Standard requirement:** "Process knowledge assertions MAY be shared across organizational units as investigation priorities. Calibrated evidence SHALL NOT be assumed transferable without local verification. Measurement system evidence MAY transfer between identical equipment with mandatory local installation verification."

### Contradiction as System Health

Disagreement about the graph is healthy. A graph where no one disagrees is either trivially simple or no one is looking.

Contradictions — two evidence records on the same edge with conflicting conclusions — are the system working, not failing. The maturity metric isn't "zero contradictions" — it's "contradictions resolved within threshold."

**On selecting course of action** (what to DO about a contradiction): use quick decision methods — impact-difficulty matrix, multi-voting, trial. Constrained resources and time demand pragmatic selection, not consensus.

**On selecting truth for the graph** (what the EVIDENCE says): the evidence speaks. Both records exist. The posterior reflects the conflict — it's wide, uncertain. The graph IS broken at that point, and that's VISIBLE. Resolution comes from further investigation — or from accepting that the relationship is more complex than modeled (interaction term, operating region constraint).

Forced failure testing is specifically powerful here — it produces clear evidence of the graph STRUCTURE itself breaking, not just the model. If you inject a failure and the detection system misses it, the graph's detection edge is wrong. That's unambiguous.

**Standard requirement:** "The organization SHALL treat contradictions in process knowledge as investigation opportunities, not failures. Contradictory evidence SHALL be retained and visible. Resolution SHALL be through further evidence gathering, not through deletion of inconvenient data. A Level 3 organization demonstrates that contradictions are resolved within its configured threshold, not that contradictions never occur."

### Unified Knowledge Views — Same Axis

QFD, FMIS, control plan, and process plan are NOT separate documents. They are views on ONE knowledge structure. The same nodes and edges, filtered and rendered differently.

| View | What it shows | Filter on knowledge structure |
|------|-------------|------------------------------|
| QFD | Customer requirements → technical characteristics | Specification nodes + relationship edges to process parameters |
| FMIS | Failure modes with causes and effects | Failure mode nodes + upstream cause edges + downstream effect edges |
| Control Plan | What to monitor, how, how often, what to do | Monitored nodes + measurement edges + reaction plans |
| Process Plan | Operations in sequence with parameters | Process parameter nodes in operation sequence |

Edit one view, the others update. Add a cause to an FMIS row, the process plan gains a parameter. Link a gage to a control plan item, the equipment view gains a measurement edge.

**QFD versioning — forward-only propagation:**

The QFD cascade flows left to right: Customer Requirements → Product Characteristics → Process Characteristics → Process Controls → Quality Controls. This is the classic four-house structure.

**Forward propagation:** If the customer changes a specification, that change propagates rightward. New process parameters may be needed. New control plan items may be required. The FMIS rows connected to the changed specification get flagged for review.

**No backward propagation:** If you discover a better process method (houses 3-4), you CANNOT retroactively change the product specification (houses 1-2). That's a new product or process version. The QFD version increments. The old version persists as history.

This prevents the dangerous practice of adjusting specs to fit the process. The customer defined the requirement. The process must meet it. If it can't, that's a signal for investigation — not a reason to change the spec quietly.

**Standard requirement:** "The organization SHALL maintain process knowledge views (risk assessment, control plan, process plan) as derived representations of a single knowledge structure, not as independent documents. Customer requirements SHALL propagate forward through the knowledge structure. Process changes SHALL NOT alter product specifications without explicit product version change."

### Detection Mechanism Hierarchy

AIAG's 1-10 detection scale mixes mechanism and likelihood into one number. OLR-001 separates WHAT the detection mechanism is from HOW WELL it works. The mechanism is the ladder level. The effectiveness is the evidence from calibration.

**The Ladder:**

| Level | Mechanism | Independence | Example | Calibration method |
|-------|-----------|-------------|---------|-------------------|
| 1 | **Source prevention** | Physics | Fixture won't accept wrong part | Verify on installation + after modification |
| 2 | **Automatic arrest** | Physics + sensor | Vibration sensor de-energizes spindle | Verify on installation + after modification |
| 3 | **Automatic detection + segregation** | Sensor + automation | Vision system rejects to scrap bin | Forced failure test: inject known defects |
| 4 | **Automatic alert + human decision** | Sensor + human | SPC alarm → operator decides | Process confirmation: observe response |
| 5 | **Structured human inspection** | Procedure + human | Operator checks every 10th part per control plan | Process confirmation: observe compliance |
| 6 | **Unstructured observation** | Human only | Operator notices surface looks rough | Not reliably calibratable |
| 7 | **Downstream detection** | Next process | Assembly finds part doesn't fit | Not a detection system — a failure |
| 8 | **Undetectable** | None | Escapes to field | Not detectable — invest to move up |

**Minimum level per classification tier:**
- **Critical characteristics:** Level 4 or above. No critical characteristic may rely solely on unstructured human observation.
- **Major characteristics:** Level 5 or above.
- **Minor characteristics:** Any level.

A Level 3 organization SHALL have no critical characteristics at Level 6 or below. This is auditable — the auditor checks each critical characteristic's detection mechanism level against the minimum.

**Investment direction:** The organization SHOULD move critical characteristics UP the ladder over time. Going from Level 5 (structured inspection) to Level 3 (automated detection) is a measurable improvement in detection capability. The health metric tracks the distribution: "what % of critical characteristics are at each level?"

**Standard requirement:** "The organization SHALL classify detection mechanisms by their independence from human judgment using the detection mechanism hierarchy. Critical characteristics SHALL employ detection at Level 4 or above. The organization SHALL track detection mechanism distribution and demonstrate upward movement on critical characteristics over time."

### Equipment and TPM+

Equipment is a knowledge domain, not just an asset register. The press, the oven, the CMM — each is a node in the knowledge structure. Its reliability, capability, and condition are edges calibrated by evidence.

**Equipment reliability as knowledge:** MTBF, failure patterns, Weibull parameters — these are calibrated edges. "This press has a characteristic life of 2,400 hours with shape parameter 2.1" is an evidence-based assertion about the equipment→process_stability relationship. When the press gets rebuilt, that evidence is stale — recalibrate.

**Measurement system capability:** Gage R&R results calibrate the measurement→quality_characteristic edge. If your measurement system contributes 30% of total variation, every evidence record gathered through that system carries an uncertainty premium. The graph knows this — measurement capability affects confidence in everything downstream.

**TPM+ practices:**

Pre-operation checklists: <8 items, prompt-response format. "Coolant level OK? Guard in place? Air pressure in range?" These are process confirmations scoped to equipment. Each completed checklist is evidence that the equipment was in known state before the shift.

T-card / kamishibai: Visual management boards with cards representing verification tasks. Cards flip when completed. A glance shows what's been verified today. Each card flip IS a verification record — the paper equivalent of a process confirmation timestamp.

Aerospace turnaround model: Before each cycle (shift, batch, changeover), verify the system is in the state the knowledge structure expects. This is the pre-flight checklist applied to manufacturing. The checklist items reference knowledge structure elements — "verify parameter X is at setpoint Y per FMIS row #Z."

**Checklists as forced failure pathway:** If the checklist says "verify emergency stop functions" and the operator tests it — that's a forced failure test on the safety detection edge. The checklist item doubles as an FFT form. One action, two evidence types.

**Standard requirement:** "The organization SHALL maintain equipment as knowledge domain elements with calibrated reliability and capability evidence. Pre-operation verification SHALL confirm equipment is in a known state consistent with the knowledge structure. Checklist-based verification MAY serve as both process confirmation and forced failure test evidence where applicable."

### Fish Market and Practitioner-Initiated Assertion Capture

Place yesterday's defects on a table. Physical parts. Visible. Experts walk by, inspect, write assertions on cards.

"I think this is a tooling wear issue" → assertion: tooling_wear → surface_defect (FMIS row, attributed, dated)
"This looks like material contamination" → assertion: contamination → inclusion_defect (FMIS row, attributed, dated)

The fish market is a Gemba signal source in physical form. Each card is a knowledge contribution. The assertions enter the system for investigation. The expert's experience is valued, recorded, and subjected to rigor — not dismissed.

Not prescribed. Recommended as a practice that organizations SHOULD consider for enabling the cultural inversion (§17).

### Paper Implementation Proof

The standard must be implementable without software. Here is the complete paper implementation:

**The "graph" is an FMIS binder.** Each row is a relationship assertion. Columns: cause, failure mode, effect, evidence reference, confidence (H/M/L), last validated date, classification tier (critical/major/minor), calibrated (Y/N).

**The forms library:**
- QFD matrix (House of Quality) — seeds FMIS rows from customer requirements
- 3P process design worksheet — documents process steps and parameter relationships
- DOE planning and analysis form — calibrates causal edges with effect sizes
- Gage R&R study form — calibrates measurement edges
- SPC chart template — monitors node distributions, flags staleness
- Process confirmation checklist — verifies standardized behavior on the floor
- Forced failure test form — calibrates detection edges
- Investigation summary form — documents new relationships and evidence
- Control plan form — references FMIS row numbers for each monitored characteristic

**Every form has a field:** "Update FMIS row #___" — that's the writeback. That's the loop closing.

**Knowledge health is countable by hand:**
- Calibration rate: count calibrated rows / total rows
- Staleness rate: count rows past their review date / total calibrated rows
- Gap ratio: count assertion-only rows / total rows
- Proactive/reactive: count internal signals before customer signals / total customer-facing signals

A shop with zero software, one binder, and a forms library can achieve Level 3.

Level 4 (predictive) requires arithmetic on effect sizes. For a small graph, a calculator suffices. For a large graph, software is practical but not mandatory.

**The standard is real. SVEND just makes it fast.**

## Standard Structure (for drafter)

### Layer 1: Requirements (the standard itself)

Auditable with a clipboard. No software required.

```
OLR-001: ORGANIZATIONAL LEARNING RATE STANDARD

1. Scope and Purpose
   1.1 What this standard covers
   1.2 What it does not cover
   1.3 What this standard REJECTS from contemporary QMS
   1.4 Relationship to ISO 9001, IATF 16949, AS9100D

2. Terms and Definitions

3. The Three Concerns
   3.1 Process Knowledge — structured, evidence-based, maintained
   3.2 Learning System — detect → investigate → standardize → verify
   3.3 Compliance — audit evidence and regulatory mappings
   3.4 How the three concerns interact (peers, not hierarchy)

4. Process Knowledge Requirements
   4.1 Structured representation of process relationships
   4.2 Evidence provenance on claimed relationships
   4.3 Quantified confidence (minimum: high/medium/low + date + source)
   4.4 Knowledge gap visibility
   4.5 Knowledge maintenance (staleness recognition, contradiction investigation)
   4.6 Growth from problems (iterative, not comprehensive upfront)
   4.7 Node classification tiers (critical/major/minor)
   4.8 Classification determines evidence minimum and staleness threshold

5. Unified Knowledge Views
   5.1 QFD, FMIS, control plan, process plan as views on ONE knowledge structure
   5.2 QFD view — customer requirements → technical characteristics (specification nodes)
   5.3 FMIS view — failure modes with causes and effects (failure mode nodes + edges)
   5.4 Control plan view — monitored characteristics with methods and reaction plans
   5.5 Process plan view — operations in sequence with parameters
   5.6 Edit one view, others update — single source of truth
   5.7 QFD versioning and forward-only propagation rule:
       - Customer requirements propagate forward to process/quality characteristics
       - Process improvements do NOT propagate backward to product specifications
       - Changing specifications = new product/process version

6. Pre-Production Knowledge Design
   6.1 Customer requirements translation (QFD → classified knowledge elements)
   6.2 Process design and knowledge structure (3P → nodes and relationship assertions)
   6.3 Physical validation (moonshining → evidence from builds, graph calibration)
   6.4 Special process qualification (formal testing → high-strength evidence)
   6.5 Configuration boundaries (product variants mapped to process knowledge)
   6.6 First article verification (full-chain validation, remaining edges calibrated)
   6.7 Control plan derivation from validated knowledge structure

7. Learning System Requirements
   7.1 Signal detection — unified event capture from all sources
   7.2 Investigation — structured methodology producing evidence
   7.3 Standardization — encode conclusions as controlled artifacts
   7.4 Verification — confirm standards hold in practice
       7.4.1 Process confirmation (direct observation)
       7.4.2 Forced failure testing (empirical detection verification)
   7.5 Knowledge feedback — investigation conclusions update process knowledge

8. Evidence Requirements
   8.1 What constitutes evidence — externalized, traceable, independently verifiable
   8.2 The calibration line: evidence record exists = calibrated, none = assertion
   8.3 Evidence strength (controlled experiment > study > structured observation > any record)
   8.4 Provenance — every evidence record traceable to source, date, method, FMIS row
   8.5 Recency — more recent evidence carries more weight (mechanism in Annex C)
   8.6 Inherited evidence (customer/supplier engineering data as calibration source)
   8.7 Minimum calibration expectations per maturity level AND per node tier

9. Detection Mechanism Hierarchy
   9.1 The ladder — ranked by independence from human judgment:
       Level 1: Source prevention (process cannot produce defect — poka-yoke at source)
       Level 2: Automatic arrest (process stops on defect — poka-yoke at detection)
       Level 3: Automatic detection + segregation (vision system, automated inspection)
       Level 4: Automatic alert + human decision (SPC alarm, andon)
       Level 5: Structured human inspection (planned check per control plan)
       Level 6: Unstructured human observation (operator notices during handling)
       Level 7: Downstream detection (next process or customer finds it)
       Level 8: Undetectable by current methods (escapes to field)
   9.2 Each level has different evidence profile for calibration:
       Level 1-2: Binary verification (one test), near-permanent evidence
       Level 3: Periodic FFT (inject known defects, measure detection rate)
       Level 4-5: Process confirmation (observe compliance + response)
       Level 6-8: Not calibratable — investment target to move UP the ladder
   9.3 Minimum mechanism level per node classification tier:
       Critical characteristics: Level 4 or above (no reliance on unstructured observation)
       Major characteristics: Level 5 or above
       Minor characteristics: any level
   9.4 Calibration frequency determined by mechanism level:
       Level 1-2: On installation + after modification
       Level 3: Per configured FFT schedule
       Level 4-5: Per configured PC schedule
   9.5 Investment direction: move critical characteristics UP the ladder over time

10. Risk Assessment Requirements
    10.1 Risk as a property of process knowledge (not a separate document)
    10.2 Failure modes as knowledge elements with upstream causes
    10.3 Severity from evidence (effect edge posteriors), not committee vote
    10.4 Occurrence from evidence (cause→failure_mode edge posteriors)
    10.5 Detection from mechanism level (§9) + calibration evidence
    10.6 Manual assessment permitted where empirical data unavailable
    10.7 Classification tier determines minimum acceptable evidence for risk scores

11. Equipment and Measurement Systems
    11.1 Equipment as knowledge domain — reliability, capability, condition
    11.2 Equipment reliability as calibrated edge (MTBF, failure patterns, Weibull)
    11.3 Measurement system capability (Gage R&R as edge calibration)
    11.4 TPM+ practices:
        11.4.1 Pre-operation checklists (<8 items, prompt-response — PC scoped to equipment)
        11.4.2 T-card / kamishibai systems for routine verification
        11.4.3 Aerospace turnaround model — structured verification of known state before stress
    11.5 Checklist items as forced failure test pathway (verify E-stop = FFT on safety edge)
    11.6 Maintenance events as staleness triggers on downstream edges
    11.7 Calibration program linked to measurement nodes in knowledge structure

12. Customer Satisfaction
    12.1 Customer-facing knowledge elements identified (from QFD)
    12.2 Monitoring of customer-facing elements (SPC, PC, FFT)
    12.3 Proactive/reactive detection ratio as primary satisfaction metric
    12.4 Customer feedback as signal source mapped to knowledge elements
    12.5 Signal resolution velocity on customer-facing elements
    12.6 Scope: product and service quality. Commercial satisfaction excluded.

13. Process Knowledge Health Metrics
    13.1 Calibration rate (% of relationships with empirical evidence)
    13.2 Staleness rate (% of relationships past review cycle)
    13.3 Contradiction rate (% with conflicting evidence)
    13.4 Signal resolution velocity (time from signal to knowledge update)
    13.5 Knowledge gap ratio (identified gaps / total relationships)
    13.6 Proactive/reactive ratio on customer-facing elements
    13.7 Detection mechanism distribution (% of critical chars at each ladder level)
    13.8 How to compute these with and without software

14. Maturity Levels
    14.1 Level 1: Structured
    14.2 Level 2: Learning
    14.3 Level 3: Sustaining
    14.4 Level 4: Predictive (proportional to process complexity)
    14.5 Level transitions — up and down
    14.6 What the auditor looks for at each level

15. Competency as Continuous Practice
    15.1 The See-Do-Teach model:
        Stage 1 SEE: observe practice performed by competent person, describe what was observed
        Stage 2 DO: execute under controlled/drill conditions, evaluator verifies method + result
        Stage 3 TEACH: demonstrate mastery by teaching — student's Stage 2 success = your evidence
    15.2 Competency is never assumed permanently — continuous demonstration through practice
    15.3 The work IS the training:
        - Process confirmation = competency demonstration + evidence production + observation capture
        - Forced failure drill = skill verification + detection calibration
        - Gemba walk = leadership practice + signal harvesting
    15.4 Checklists liberate cognition — routine is externalized so the brain is free to notice anomalies
    15.5 Competency evidence = practice records (PCs conducted, FFTs executed, assertions contributed)
    15.6 No separate training matrix needed — FMIS evidence IS the competency record
    15.7 Minimum competency per maturity level:
        Level 1: key personnel Stage 1 complete for FMIS + signals
        Level 2: investigation leads Stage 2 complete for DOE + evidence collection
        Level 3: process owners Stage 3 complete — can teach the system
        Level 4: technical staff can execute + teach quantitative methods
    15.8 Paper implementation: competency card per person (practice, stage, evidence ref)

16. Leadership and Resource Requirements
    16.1 Configuration as quality policy and commitment contract
    16.2 RACI derived from configuration (owner, accountable, consulted, informed)
    16.3 Continuous visibility into process knowledge health
    16.4 Resource allocation informed by knowledge gaps
    16.5 Decision points (replaces periodic management review)

17. Controlled Documents and Records
    17.1 Documents encode process knowledge — linked to FMIS rows they govern
    16.2 Review triggered by knowledge change, not calendar
    16.3 Approval as knowledge commitment — "this encodes our current understanding"
    16.4 Version history traces to knowledge events (investigation #, evidence change)
    16.5 Obsolescence determined by knowledge state, not administrative decision
    16.6 Record retention tied to process relevance, not arbitrary periods
    16.7 Paper implementation: cover sheet listing governed FMIS rows

18. Organizational Culture and Knowledge Capture
    18.1 The cultural inversion — knowledge flows in all directions
    17.2 Continuous capture from all personnel (not periodic, not exit-only)
    17.3 Gemba as knowledge harvesting (leader walks produce FMIS assertions)
    17.4 Fish market / defect exhibitions (practitioner-initiated assertion capture)
    17.5 Attribution and respect for contributed assertions
    17.6 Knowledge persistence through personnel transitions

19. Staleness and Process Change
    19.1 Explicit staleness triggers (material, equipment, personnel, environment, engineering)
    18.2 Time-based thresholds as secondary detection
    18.3 Review protocol for triggered edges
    18.4 Staleness vs invalidation — the edge is suspect, not automatically wrong

20. Multi-Site and Organizational Change
    19.1 Assertion transfer (yes) vs evidence transfer (no)
    19.2 Measurement system evidence exception (identical equipment, local verification)
    19.3 Merger, acquisition, divestiture — graph persistence at organizational unit
    19.4 Local verification requirement

21. Contradiction Management
    20.1 Contradictions as system health indicator — disagreement is healthy
    20.2 Both evidence records retained and visible
    20.3 Resolution through further evidence, not deletion
    20.4 Decision methods for course of action vs truth determination
    20.5 Forced failure testing as graph structure validation

22. Supplier Knowledge Integration
    21.1 Supplier evidence as knowledge input (inherited calibration)
    21.2 Supplier claims as signal source
    21.3 Supplier response quality assessment
    21.4 Incoming material properties as node evidence

23. Compliance Mappings
    22.1 ISO 9001:2015 clause-by-clause mapping
    22.2 IATF 16949:2016 additional requirements mapping
    22.3 AS9100D additional requirements mapping
    22.4 How OLR-001 Level 3 satisfies ISO 9001 certification requirements

24. Audit Protocol
    23.1 What the auditor inspects (knowledge structure, evidence, health metrics)
    23.2 Evidence sampling methodology
    23.3 Maturity level assessment procedure
    23.4 Certification reflects current state, not annual event
    23.5 Paper vs software implementation — both auditable

25. Paper Implementation
    24.1 FMIS binder as the knowledge structure
    24.2 Forms library (QFD, 3P, DOE, Gage R&R, SPC, PC, FFT, investigation, checklist)
    24.3 "Update FMIS row #___" — the paper writeback mechanism
    24.4 Manual health metric computation
    24.5 What Level 4 requires beyond paper (arithmetic at minimum)
```

### Layer 2: Technical Annexes (computational rigor)

Optional for Levels 1-3. Required for Level 4. This is where SVEND's implementation shines — but the annexes are implementable by anyone.

```
Annex A: Process Knowledge Graph Schema
   - Node types, edge types, evidence record structure
   - Minimum viable schema (spreadsheet-compatible)
   - Full schema (software implementation)

Annex B: Bayesian Evidence Weighting
   - Prior and posterior computation
   - Evidence strength scoring
   - Recency decay functions
   - Worked examples

Annex C: Process Knowledge Entropy
   - Formal entropy definition
   - Computation from graph metrics
   - Simplified computation from manual counts
   - Interpretation and thresholds

Annex D: Closed-Loop Learning System Reference Architecture
   - Signal taxonomy and routing
   - Investigation methodology
   - Standardization workflow
   - Verification protocols

Annex E: Compliance Mapping Tables
   - ISO 9001:2015 full mapping (every clause → OLR-001 section)
   - IATF 16949:2016 full mapping
   - AS9100D full mapping

Annex F: Glossary
```

## Reference Documents

The drafter should read these in order:

1. `docs/planning/object_271/identity.md` — product identity (process knowledge system)
2. `docs/standards/GRAPH-001.md` — the graph spec (schema, evidence, gaps, Synara)
3. `docs/standards/LOOP-001.md` — the loop spec (signals, investigation, standardize, verify)
4. `docs/planning/object_271/edges.md` — Edge 6 (certification concept)
5. `docs/planning/object_271/conference.md` — D11 (Airbus model, identity decisions)
6. `docs/planning/object_271/enterprise_configuration_spec.md` — presets and settings
7. `docs/planning/object_271/configuration_service_spec.md` — scoping and tiers

## Naming Decision Needed

| Option | Full Name | Vibe |
|--------|-----------|------|
| OLR-001 | Organizational Learning Rate | Measures what matters — learning speed |
| PKS-001 | Process Knowledge Standard | Descriptive, boring, clear |
| CALS-001 | Calibrated Assurance Level Standard | Nods to calibration as core concept |
| EPK-001 | Evidence-Based Process Knowledge | The thesis in the name |
| CCS-001 | Continuous Calibration Standard | Echoes CCS 1948 lineage |

Eric to decide.

## Constraints for Drafter

1. **Platform independence is non-negotiable.** Every Layer 1 requirement must be satisfiable without software. The standard never names a tool, screen, API, or vendor. Test every requirement: "could someone do this with a whiteboard and a filing cabinet?" If no, it belongs in an Annex, not in the requirement.

2. **ISO/IATF/AS9100D compliance mappings must be complete and defensible.** An org certified to OLR-001 Level 3 should automatically satisfy ISO 9001 requirements. The mapping tables in Annex E must be clause-by-clause with explicit cross-references. An ISO auditor reading the mapping should say "yes, this covers my clause."

3. **The math belongs in Annexes, not in requirements.** Layer 1 says "quantified confidence." Annex B says "Bayesian posterior with Beta-Binomial conjugate prior." Layer 1 says "knowledge health improving." Annex C says "Shannon entropy decreasing at rate R." An org at Level 1-3 never needs to open the Annexes.

4. **Maturity levels must be auditable by observation.** The auditor walks the floor, looks at the process knowledge structure (whatever form it takes), samples evidence records, checks dates, asks "what don't you know?" and "what changed since last time?" That's the audit. No login required.

5. **Tone: authoritative, practitioner-authored, not bureaucratic.** This is a standard written by people who've stood in the circle on the factory floor. It should read like it. ISO's passive voice and committee-speak are not the model. Clear, direct, opinionated where it matters.

6. **The standard must define what it REJECTS from contemporary QMS, not just what it adds.** Section 1 or 3 should explicitly state: "This standard does not require a CAPA process, a standalone FMEA document, periodic management reviews, or role-based training matrices. It replaces these with [the OLR-001 equivalents]. Organizations maintaining these artifacts for regulatory compatibility may generate them from the Learning System's outputs."

## Pressure Testing Needed

Before Cowork drafts, these questions need answers:

1. **What is the minimum viable process knowledge structure at Level 1?** RESOLVED: An FMIS (Failure Modes Investigation System) with deduplicated entities, directional relationships (cause → failure mode → effect), and visible distinction between assertion and evidence. A standard FMEA table is NOT sufficient — it lacks entity uniqueness and evidence provenance. But an FMIS plus a cross-referenced forms library (DOE, Gage R&R, SPC, PC, FFT, investigation summary) IS sufficient. Each form has a "Update FMIS row #___" field — that's the paper equivalent of graph writeback. Annex A provides the paper FMIS template + forms library. The forms teach the methodology while collecting evidence — education is integrated into the standard, not separate from it.

2. **How does an auditor assess "learning rate" without longitudinal data?** RESOLVED: First certification CAN be Level 2 on a single snapshot. The test is coherence, not history. If the FMIS says "calibrated" and the auditor can trace to the evidence source (DOE form, study, investigation) and the numbers match — that's Level 2. Auditor samples 5 calibrated rows, traces each to evidence, verifies coherence. History (rate of change between audits) becomes relevant for Level 3 — sustaining requires demonstrating that knowledge health is MAINTAINED over time, which inherently needs two or more assessment points.

3. **What's the minimum evidence bar for "calibrated"?** Is an operator saying "yeah, temperature affects viscosity" evidence? Or does calibrated strictly mean "measured effect size from controlled experiment"? The standard needs a clear hierarchy, and the threshold for "calibrated vs assertion" must be explicit.

4. **Can a single-product, single-process organization reach Level 4?** RESOLVED: Yes. Level 4 is about predictive capability proportional to process complexity, not about graph size. A 10-step single-product shop with 50 calibrated edges that can predict "temperature up 5°C → defect rate increase 2.3% ± 0.8%" and verify it — that's Level 4. That's arithmetic, not Monte Carlo. A small calibrated graph where predictions validate is more mature than a large uncalibrated one. The standard says: "demonstrates predictive capability proportional to the complexity of their process." Small shops predict by hand. Enterprises use computational tools (Annex C). Both are Level 4.

5. **How do the compliance mappings handle ISO clauses that have no OLR-001 equivalent?** RESOLVED: Full clause-by-clause pressure test completed. Results:

   **Direct/strong mapping (~85%):** Clauses 4 (context = graph + signals + configuration), 5 (leadership = configuration-as-commitment-contract + RACI), 6 (planning = FMIS + gap report + maturity targets), 7.1.5-7.1.6 (measurement + organizational knowledge), 7.2 (competence from graph gaps), 7.5 (documented information), 8.1 (control plans), 8.4 (supplier management), 8.5 (production control), 8.7 (nonconforming = signals), 9.1.1/9.1.3 (monitoring = SPC + graph metrics), 9.2 (audit), 9.3 (management review = continuous dashboard), 10 (improvement = the entire Loop).

   **Resolved gaps:**
   - **8.2/8.3 (Requirements review + Design):** 3P (Production Preparation Process) with QFD. Customer requirements translate to specification nodes via QFD. Moonshining cycles are investigations that calibrate the graph before production. 3P IS the Loop running pre-production. Native to the standard, not supplementary.
   - **9.1.2 (Customer satisfaction):** Proactive/reactive ratio on customer-facing graph nodes. Customer-facing nodes identified via QFD. Monitored via SPC + process confirmation. The metric: what % of quality issues affecting customers were detected by YOUR monitoring vs reported BY the customer? Computed from existing operational data, unfakeable, auditable. Product satisfaction = customer-facing node health. Service satisfaction = customer-signal resolution velocity. Commercial satisfaction (price, terms) explicitly out of scope.
   - **8.6 (Product release):** Release is a READ operation on the knowledge structure: are customer-facing nodes in control, critical detection mechanisms calibrated, relevant edges non-stale? Risk assessment branches off QFD cascade as annotation, not gate. Release justification lives on the knowledge structure — the graph already contains capability, detection level, and process state for every customer-facing characteristic. No separate release form needed.

   **Remaining supplementary (< 5%):** 7.4 general communication processes (OLR covers quality communication via signals but not general business communication). Acknowledged as out of scope — OLR is a process knowledge standard, not a business communication standard.

   **Key design decisions from this analysis:**
   - Configuration = Quality Policy + Commitment Contract + RACI Matrix. One surface, three ISO clauses (5.1, 5.2, 5.3).
   - The configuration printout IS the quality policy document.
   - Each policy setting has owner (R), accountable party (A), consulted (signal routing), informed (notification settings).
   - Quality objectives = maturity level targets + knowledge health targets set in configuration.
   - 3P/QFD/moonshining are native to the standard under "Pre-Production Knowledge Building" — not an afterthought.
   - Customer satisfaction is measured by process data, not surveys. Proactive/reactive ratio is the primary metric.

These should be resolved in the brief or flagged for Eric's decision before drafting.
