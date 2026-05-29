# TRIZ Analysis: SVEND forgeclaude Architecture

**Date:** 2026-05-03
**Technique:** TRIZ Contradiction Analysis + Inventive Principles
**Context:** Post-focus-group identity pivot. SVEND = computation + documentation. Claude = facilitator. PROVA/PCL = Claude's invisible memory. Need to design forgeclaude integration layer and reduce the overall system to its irreducible form.

---

## Contradictions

### Contradiction 1 (PRIMARY): Knowledge Depth vs. User Simplicity

The system must accumulate deep institutional knowledge (graph, Bayesian updating, evidence discrimination, propagation) while users experience a simple conversation.

- **Improving:** Information loss (#24) — retain institutional knowledge
- **Degrading:** Ease of operation (#33) — simple conversation experience
- **Principles:** 32 (Color changes), 28 (Mechanics substitution), 2 (Taking out), 13 (The other way around)

### Contradiction 2: Integration Power vs. Maintenance Burden

forgeclaude must give Claude broad, safe access to many forge packages — without becoming another heavy API surface to maintain.

- **Improving:** Adaptability (#35)
- **Degrading:** Complexity of device (#36)
- **Principles:** 15 (Dynamics), 29 (Pneumatics/hydraulics), 37 (Thermal expansion), 28 (Mechanics substitution)

### Contradiction 3: Knowledge Persistence vs. Ephemeral Context

Knowledge must outlive sessions, but Claude's context is inherently ephemeral.

- **Improving:** Duration of action (#15)
- **Degrading:** Loss of substance (#23)
- **Principles:** 35 (Parameter changes), 28 (Mechanics substitution), 34 (Discarding/recovering), 3 (Local quality)

### Contradiction 4: Expert Judgment vs. Approachability

System must deliver expert-quality OpEx implementation to the 99% who can't implement it alone.

- **Improving:** Productivity (#39)
- **Degrading:** Ease of operation (#33)
- **Principles:** 28 (Mechanics substitution), 15 (Dynamics), 17 (Another dimension), 10 (Preliminary action)

---

## Solutions by Principle

### Principle 2 — Taking Out
Separate "learning why" from "doing." Claude acts and records but never explains methodology during execution — only surfaces reasoning if asked "why did you do that?" OpEx knowledge is present but physically removed from conversation flow, stored in PROVA edges and PCL annotations, invisible unless pulled. User sees: "Your defect rate driver is moisture content in Bay 3. Here's the control chart. Want me to set up monitoring?" The Shewhart theory, Bayesian updating, FMEA linkage — all executed but never spoken aloud unprompted.

### Principle 3 — Local Quality
Each PCL measure carries "explanation density" metadata — a per-characteristic setting controlling how much Claude elaborates. A measure the user created yesterday gets minimal narration ("still in control"). A first out-of-control signal gets full guided explanation. A measure the user has never interacted with gets a one-sentence contextual introduction. The system is locally adapted to the user's familiarity with each specific process characteristic, not uniformly simple or uniformly deep.

### Principle 10 — Preliminary Action
On first data source connection, forgeclaude pre-runs a silent "intake analysis": capability indices, normality checks, FMEA risk rankings, control chart violations. Results sit in PROVA as pre-computed theories at zero confidence. When the user asks "what's wrong with Line 4?", Claude retrieves and validates rather than computing from scratch — instant insight, not diagnostic process. Expert judgment happened before the user needed it.

### Principle 13 — The Other Way Around
Instead of Claude explaining OpEx to users, make users teach Claude about their process. "Walk me through what happens between the press and the oven." User narrates. Claude silently maps to VSM, identifies wastes, populates PCL from casual language ("it sits there for like twenty minutes" becomes a WIP wait-time measure). User thinks they're onboarding a new employee. Claude is conducting a gemba walk. Knowledge capture happens because the user thinks they're the expert — which they are, about their own process.

### Principle 15 — Dynamics
forgeclaude uses runtime capability discovery. Each forge package registers capabilities at import via manifest (`forgespc.CAPABILITIES = ["xbar_r", "cusum", "ewma", ...]`). forgeclaude doesn't hardcode packages. New packages auto-register. Integration layer is a protocol, not a catalog. Maintenance burden drops to zero for adding new computation.

### Principle 17 — Another Dimension
Add a temporal dimension. Claude timestamps question patterns, builds a "maturity trajectory" in PROVA. After 90 days, adjusts vocabulary, explanation depth, proactive suggestions. User never configures skill level. They notice Claude "gets them" better over time. No competitor can copy this — requires persistent memory across months.

### Principle 28 — Mechanics Substitution
Replace forgeclaude's procedural Python with a declarative contract system. Each computation described as a JSON schema contract (inputs, outputs, pre/postconditions). forgeclaude becomes a contract executor + contract directory. New analysis type = new JSON file, not new code. Claude writes contract requests, not Python.

### Principle 29 — Pneumatics/Hydraulics
Replace rigid graph with fluid gradient fields. Knowledge as continuous confidence scores flowing between measures. System observes moisture→warping correlation automatically. No manual causal graph definition. Knowledge finds its own paths like pressure through a hydraulic system.

### Principle 32 — Color Changes
Cognitive load heatmap on every response. Score sentences for jargon density, prerequisite depth. Demote sentences above user's maturity level — collapse behind toggle, lighter type, or plain-language summary. Novice sees 3 sentences, expert sees 12. Same response object, different rendering.

### Principle 34 — Discarding and Recovering
Session-end knowledge distillation. Discard full conversation, recover the delta — new theories, updated confidences, invalidated assumptions. Compress into PROVA updates. Next session: read PROVA + PCL snapshot, not old chat. Lossy compression forces signal extraction. The compression is the feature, not the bug.

### Principle 35 — Parameter Changes
Change knowledge state from explicit to implicit. "Line 4 has moisture issues" becomes: default data pull includes humidity sensors, alert thresholds tighter, FMEA pre-populates environmental factors. Knowledge manifests as system behavior, not readable propositions. Users see a system that pays attention to the right things without knowing why.

### Principle 37 — Thermal Expansion
Use tool contradictions as integration mechanism. When forgespc and forgedoe conflict, expand the contradiction into a structured decision point. "Your process is capable at current settings, but there's a better operating point requiring revalidation. Here's the cost of each path." Contradictions become the most valuable conversation moments — where real decisions live.

---

## Evaluation

### Structural Clusters

**8 distinct structural ideas** from 12 solutions:

| Cluster | Solutions | Theme |
|---|---|---|
| Knowledge Representation | P29, P35 | How knowledge is stored (gradient fields vs. implicit behavior) |
| Conversation Surface | P2, P32 | What the user sees (binary removal vs. variable rendering) |
| Temporal Accumulation | P17, P34 | What persists (maturity trajectory + lossy distillation) |
| Standalone | P3, P10, P13, P15, P28, P37 | Each structurally unique |

### Core Contradiction Engagement

**Directly addresses depth vs. simplicity:**
- P2 — spatial separation (knowledge exists but removed from flow)
- P13 — dissolves it (user provides depth unknowingly, experiences simplicity)
- P32 — variable resolution rendering
- P35 — knowledge changes state from explicit to behavioral

**Sidesteps (solves adjacent problems):**
- P3 — optimizes explanation dosing
- P10 — solves latency
- P15, P28 — solve maintainability
- P17, P34 — solve temporal persistence
- P29 — solves knowledge discovery automation
- P37 — solves internal tool disagreement

### Most Surprising Element

**P13 (The Other Way Around)** is the most surprising.

Every other solution accepts the frame: "Claude has OpEx knowledge and must deliver it palatably." P13 rejects the frame. The user is the knowledge source. Claude is the student. The "gemba walk as onboarding" inversion means:

- User feels competent (teaching), not intimidated (learning)
- Claude extracts process knowledge through natural narration, not structured forms
- VSM, waste identification, PCL population happen as side effects of conversation
- The purchase trigger (tribal knowledge retention) is satisfied by the act of onboarding itself — the user externalizes knowledge they didn't know they had

This is hard to reach conventionally because it requires abandoning the assumption that the AI is the expert. P13 positions the user as the smart one and means it.

Secondary surprise: **P37** — using tool contradictions as highest-value moments. Conventional thinking treats internal disagreement as a bug. P37 treats it as the precise moment where human judgment is irreplaceable.

### Combination Opportunities

**Combo 1: P13 + P34 + P17 = "The Accumulating Gemba Walk"**
User teaches Claude (P13). Session end distills narrative into PROVA/PCL (P34). Over months, Claude's questions get sharper — asks about gaps, not basics (P17). User experiences an "employee" who is getting better at the job. Tribal knowledge retention made literal.

**Combo 2: P3 + P32 + P17 = "Adaptive Resolution"**
Per-measure explanation density (P3) + cognitive load rendering (P32) + maturity trajectory (P17). Three independent adaptation axes, none requiring user configuration.

**Combo 3: P10 + P29 + P37 = "Pre-computed Tension"**
Silent intake analysis (P10) + automatic correlation discovery (P29) + contradiction surfacing (P37). User never ran an analysis. System presents: "Your capability data says X, but your FMEA ranking implies Y. Which do you trust more?"

**Combo 4: P15 + P28 = "Self-Describing Computation Layer"**
Runtime discovery (P15) + declarative contracts (P28). forgeclaude = contract directory that discovers its own contents at startup. New forge package = JSON manifest + Python module. No forgeclaude code changes.

**Combo 5 (dangerous): P13 + P35 = "Invisible Knowledge Transfer"**
User teaches Claude (P13). Knowledge manifests as behavior, never explicit statements (P35). User taught Claude once. Claude never repeats what it learned. Just acts differently. Deepest form of institutional knowledge — how experienced operators actually carry knowledge (habits, not propositions). Risk: user can't see what the system "knows" to correct it when wrong.

### Two Philosophies

The solutions split into two fundamentally different philosophies:

**Philosophy A** (P2, P3, P32): Information hiding. System has knowledge, controls how much leaks into conversation. Conventional, implementable, incremental.

**Philosophy B** (P13, P35, P29): Knowledge metabolism. System acquires knowledge from user and manifests it as behavior, not explanation. Unconventional, harder to build, but addresses tribal knowledge retention at a structural level.

The most interesting territory is the boundary between these philosophies — where the system simultaneously learns from the user (B) and selectively reveals what it's learned (A), with P37's contradiction-surfacing as the mechanism keeping the user in control.
