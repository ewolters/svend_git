# Morphological Analysis: forgeclaude Architecture

**Date:** 2026-05-03
**Technique:** Morphological Analysis (Zwicky Box)
**Context:** Post-identity-pivot. SVEND = computation + documentation, Claude = facilitator, PROVA/PCL = invisible memory. Designing forgeclaude integration layer + plugin surface. Follows TRIZ analysis of the same problem.

---

## Morphological Box

| Dimension | Option 1 | Option 2 | Option 3 | Option 4 | Option 5 |
|---|---|---|---|---|---|
| **Computation Boundary** | Thin Wrapper (1:1 function wrappers) | Intent Functions (high-level orchestration) | Declarative Spec (JSON spec + planner) | Conversational REPL (NL + dispatch) | Pipeline Builder (stage assembly) |
| **Knowledge Persistence** | Eager Sync (every turn read/write) | Session Snapshot (load/flush) | Event Sourced (immutable log, replay) | Pull-on-Demand (lazy read, immediate write) | Dual-Ring (fast working + slow institutional) |
| **Audit Trail** | Inline Narration (auto reasoning paragraphs) | Post-Hoc Reconstruction (rebuild on demand) | Dual-Track (structured log + human summary) | Auditor-as-User (auditor gets own Claude session) | Continuous Ledger (append-only chained log) |
| **Failure Semantics** | Traffic Light (red/yellow/green) | Socratic Surfacing (ask questions to resolve) | Full Distribution (always return posteriors) | Silent Degradation (best answer, threshold for surfacing) | Failure as Data (failures become PROVA evidence) |
| **Plugin Grain** | One Giant Tool (single execute) | Domain Cluster (one per domain, 3-8 tools) | Atomic Primitives (50-100+ tools) | Workflow Tools (by user workflow) | Two-Phase Discovery (catalog + dynamic load) |
| **User Adaptation** | Explicit Levels (beginner/practitioner/expert) | Inferred Calibration (learn from patterns) | Always Teach (introductory + expandable) | Challenge-Response (calibration questions) | No Hiding (show everything, formatting only) |

**Total design space:** 15,625 combinations

---

## Random Combinations Evaluated

### Combination 1: Thin Wrapper + Eager Sync + Dual-Track + Socratic Surfacing + Domain Cluster + Inferred Calibration

**Feasible?** With modification (throttle Eager Sync to dirty-flag writes only).

**Interesting:** Socratic Surfacing + Inferred Calibration creates a feedback loop. The system asks clarifying questions when uncertain, and the *nature of the user's answers* becomes calibration signal. A practitioner answers "yeah, that's my Cpk, the gauge R&R was clean" — system learns they're experienced. A learner says "what's Cpk?" — calibration shifts. The Socratic failure mode IS the adaptation mechanism. These two dimensions collapse into one behavior.

### Combination 2: Thin Wrapper + Dual-Ring + Inline Narration + Failure as Data + Workflow Tools + Explicit Levels

**Feasible?** Yes.

**Interesting:** Failure as Data + Dual-Ring creates genuinely novel knowledge architecture. Failed capability study writes to fast ring as working context AND propagates to slow institutional ring as evidence. Over time, institutional ring accumulates failure history: "this machine has failed capability 4 times in 18 months, always on the same characteristic." **Failures are the institutional memory that matters most in manufacturing.** Explicit Levels is the weak link — should be Inferred Calibration.

### Combination 3: Thin Wrapper + Eager Sync + Post-Hoc Reconstruction + Socratic Surfacing + Two-Phase Discovery + No Hiding

**Feasible?** No. Post-Hoc Reconstruction + Eager Sync is contradictory — you already have the data, why reconstruct? No Hiding + Two-Phase Discovery = cognitive overload.

**Interesting:** Two-Phase Discovery + Socratic Surfacing has unexpected harmony. The catalog phase IS Socratic: "I found these 4 analyses that could apply. Which aspect of your process are you most concerned about?" Discovery becomes a teaching moment.

### Combination 4: Thin Wrapper + Dual-Ring + Post-Hoc Reconstruction + Failure as Data + Workflow Tools + Inferred Calibration

**Feasible?** With modification (add lightweight intent tags at each step for reconstruction).

**Interesting:** Workflow Tools + Inferred Calibration is a subtle pairing. Tools organized by workflow ("investigate capability" vs "run SPC") let Claude observe which workflows the user gravitates toward and infer operational maturity. Someone who jumps to "root cause analysis" is different from someone who starts with "check my data." The workflow grain CREATES the signal calibration needs.

### Combination 5: Conversational REPL + Dual-Ring + Dual-Track + Traffic Light + Domain Cluster + Challenge-Response

**Feasible?** With modification (replace Traffic Light with Silent Degradation — REPL returns rich answers, stoplight colors kill nuance).

**Interesting:** Conversational REPL + Dual-Ring is the most natural pairing in the entire set. The conversation IS the fast ring (working memory of the session), institutional knowledge is the slow ring. Each turn is ephemeral working memory, conclusions persist. The REPL metaphor maps perfectly. Everything else in this combination fights it.

### Combination 6: Declarative Spec + Event Sourced + Post-Hoc Reconstruction + Socratic Surfacing + Atomic Primitives + Explicit Levels

**Feasible?** No. Declarative Spec + Atomic Primitives is an architecture war (planner must compose 100+ primitives). Event Sourced makes Post-Hoc Reconstruction redundant.

**Interesting:** Declarative Spec + Event Sourced = **reproducible analysis for free.** Every computation is a JSON spec, every result is an immutable event. Auditor asks "why this conclusion?" — replay the event log, re-execute the specs, get identical results. Strongest audit story in the design space. Buried under bad choices elsewhere, but the core pairing is powerful.

### Combination 7: Thin Wrapper + Pull-on-Demand + Inline Narration + Full Distribution + Atomic Primitives + No Hiding

**Feasible?** No. Maximum information + maximum complexity. Learner gets Bayesian posteriors, 100 tool options, and paragraphs of reasoning. Expert workbench, not a product for the 99%.

**Interesting:** Pull-on-Demand + Inline Narration creates productive tension. Narration documents what Claude *didn't know* at each point: "I'm checking historical capability..." becomes a trace of when knowledge was pulled and what gap triggered the pull. Laziness of persistence becomes visible through narration.

### Combination 8: Declarative Spec + Eager Sync + Auditor-as-User + Failure as Data + One Giant Tool + Challenge-Response

**Feasible?** With modification (replace One Giant Tool with Domain Cluster).

**Interesting:** Auditor-as-User + Failure as Data + Declarative Spec = the most commercially differentiated audit story. Auditor gets own Claude session, asks "show me every process that failed capability in Q1." Because failures are first-class PROVA evidence AND analyses are reproducible declarative specs, auditor can re-run exact analyses. **Auditor becomes a power user, not a report reader.** Genuinely differentiated from every QMS on the market.

### Combination 9: Thin Wrapper + Dual-Ring + Dual-Track + Failure as Data + Atomic Primitives + No Hiding

**Feasible?** With modification (replace No Hiding with Inferred Calibration).

**Interesting:** Dual-Ring + Dual-Track + Failure as Data creates four distinct memory channels: fast working memory, slow institutional memory, structured audit log, human-readable narrative. Sounds over-engineered, but maps to four real audiences: operators need working context, organization needs institutional knowledge, ISO auditors need structured records, management needs readable summaries. Four audiences, four channels.

### Combination 10: Intent Functions + Eager Sync + Inline Narration + Socratic Surfacing + Atomic Primitives + Explicit Levels

**Feasible?** With modification (replace Atomic Primitives with Domain Cluster — layer conflict with Intent Functions).

**Interesting:** Intent Functions + Socratic Surfacing + Inline Narration creates "guided investigation." Claude calls `assess_capability(id)`, discovers gauge R&R data is missing, Socratic Surfacing asks "has a measurement system analysis been done?", Inline Narration documents the gap and response. **The combination accidentally reinvents the consulting engagement model.** Try → discover gap → ask → document. This is what a good consultant does on day one.

---

## Three Most Promising

### 1. Combination 2 (modified): Thin Wrapper + Dual-Ring + Inline Narration + Failure as Data + Workflow Tools + Inferred Calibration

Failure as Data + Dual-Ring is the standout insight. Manufacturing improvement IS about failures — treating them as first-class knowledge rather than error states is a structural advantage. Workflow Tools provide the right user-facing grain. Replace Explicit Levels with Inferred Calibration.

### 2. Combination 8 (modified): Declarative Spec + Eager Sync + Auditor-as-User + Failure as Data + Domain Cluster + Challenge-Response

Auditor-as-User is the most commercially differentiated idea. Every QMS generates reports for auditors. Giving auditors their own Claude session that can reproduce and interrogate analyses is "why would I use anything else." Combined with Failure as Data + Declarative Spec (reproducible analyses), the audit story is airtight.

### 3. Combination 10 (modified): Intent Functions + Eager Sync + Inline Narration + Socratic Surfacing + Domain Cluster + Inferred Calibration

The "accidental consulting model." Intent Functions + Socratic Surfacing + Inline Narration produces behavior that mimics a skilled consultant doing an assessment. Best serves "the 99% who are learners."

---

## Unexpected Dimension Interactions

### 1. Failure Semantics x Knowledge Persistence is the most consequential pairing

Failure as Data + Dual-Ring creates institutional learning. Socratic Surfacing + Inferred Calibration creates a self-calibrating teacher. Traffic Light + anything kills nuance. This pair determines whether the system is a calculator or an advisor. Failure Semantics — which sounds like error handling — is the dimension that most shapes the system's character.

### 2. Audit Trail x Computation Boundary determines reproducibility

Declarative Spec makes audits trivial (replay the spec). Thin Wrapper makes audits hard (reconstruct intent from low-level calls). Post-Hoc Reconstruction is only viable if the Computation Boundary provides enough structure. This dependency isn't obvious from the dimensions alone.

### 3. Plugin Grain x User Adaptation are secretly coupled

Atomic Primitives + No Hiding is hostile. Workflow Tools + Inferred Calibration is natural. Domain Cluster works with almost anything. The grain of tools determines how much adaptation CAN happen. Plugin Grain should be chosen after User Adaptation, not independently.

### 4. Socratic Surfacing interacts positively with almost everything

Works as: calibration signal, teaching mechanism, gap detection, audit documentation. Only failure semantic that produces MORE information. For a system targeting learners, may be the only viable choice — which would collapse that dimension.

### 5. Thin Wrapper appeared in 6 of 10 combinations and was never the interesting part

Default choice, absence of a decision. Every interesting property came from other dimensions. Suggests Computation Boundary is less consequential than it appears, or that Thin Wrapper is a local minimum — avoids risk but also avoids value. Intent Functions and Declarative Spec produced more interesting interactions.
