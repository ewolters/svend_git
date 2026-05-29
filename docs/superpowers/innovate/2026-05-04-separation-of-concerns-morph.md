# Morphological Analysis: Separation of Concerns

**Date:** 2026-05-04
**Technique:** Morphological Analysis (Zwicky Box)
**Context:** Five systems — Synara (governance), PCL (data substrate), Claude (facilitator), Workflows (orchestration), Sessions (user-facing work context). Where does each system's responsibility start and stop?

---

## Morphological Box

| Dimension | Option 1 | Option 2 | Option 3 | Option 4 | Option 5 |
|---|---|---|---|---|---|
| **Data Access Mediation** | Direct read, contract on write | Synara mediates everything | PCL as dumb bus | Tiered access by confidence | Session-scoped projections |
| **Judgment Authority** | Synara decides within rules, Claude beyond | Claude always decides, Synara constrains | User always decides | Synara decides everything | Split by reversibility |
| **Knowledge Residence** | PCL is system of record | Synara accumulates as rules | Distributed, no single source | Claude reconstructs each session | Workflow history is primary record |
| **Session-Workflow Relationship** | Sessions contain workflows | Workflows contain sessions | Same thing, different views | Fully independent | Sessions are workflows that haven't graduated |
| **Governance Scope** | Data contracts only | Data contracts plus workflow gates | Everything including Claude | Outcomes only, not actions | User-scoped governance |
| **Context Assembly** | Claude assembles its own | Session assembles context | Each system publishes fragments | Synara assembles context | Lazy pull on demand |

**Total design space:** 15,625 combinations

---

## Random Combinations Evaluated

### Combination 1
Direct read/contract on write | Synara rules + Claude beyond | Distributed knowledge | Workflows contain sessions | Data contracts + workflow gates | Session assembles context

**Feasible?** With modification (flip "workflows contain sessions" to "sessions contain workflows").

**Interesting:** "Distributed knowledge" + "session assembles context" makes the session an active knowledge integrator — a lens that pulls fragments from PCL, Synara, workflow history and composes a working view. The session becomes where understanding lives, Synara where permission lives, and nobody owns "truth" — truth is assembled on demand from evidence. This is how manufacturing tribal knowledge actually works.

### Combination 2
Direct read/contract on write | Split by reversibility | PCL is system of record | Sessions graduate to workflows | Outcomes only governance | Claude assembles its own

**Feasible?** Yes.

**Interesting:** The most natural combination. Reversibility-based judgment means low-stakes decisions happen fast, irreversible ones require Synara sign-off. Outcomes-only governance doesn't micromanage the path. Sessions graduating to workflows = Investigate → Standardize → Verify loop. **Reversibility + outcome-only governance is a trust architecture. You govern tightly where damage is permanent, loosely where it isn't, and learn from results rather than policing steps.** Mirrors how good manufacturing managers actually operate.

### Combination 3
Direct read/contract on write | Synara rules + Claude beyond | Synara accumulates as rules | Workflows contain sessions | User-scoped governance | Lazy pull on demand

**Feasible?** With modification (flip session/workflow, change user-scoped to role-scoped).

**Interesting:** "Synara accumulates as rules" + "lazy pull" creates a system where governance knowledge grows organically but is only materialized when needed. Synara isn't a gatehouse — it's a library of learned constraints consulted on demand. Lighter weight than mediate-everything. Risk: stale rules persist without confidence decay.

### Combination 4
Direct read/contract on write | Split by reversibility | Synara accumulates as rules | Sessions graduate to workflows | Outcomes only governance | Session assembles context

**Feasible?** Yes.

**Interesting:** Close cousin of Combo 2. Key swap: knowledge in Synara-as-rules instead of PCL. Synara watches outcomes, distills into rules, but rules are advisory (outcomes-only governance). Claude reads them as context for guidance, not hard constraints. **Accidentally describes how tribal knowledge works: experienced operators know rules but apply them as judgment, not law.** Session assembling context keeps Claude's role clean — facilitates with whatever context it's given.

### Combination 5
Tiered access by confidence | Split by reversibility | Distributed knowledge | Sessions contain workflows | Data contracts + workflow gates | Synara assembles context

**Feasible?** With modification (drop distributed knowledge → PCL-as-record).

**Interesting:** Tiered access by confidence = measurements with stable Bayesian confidence are freely readable, new measurements with 3 data points require acknowledgment of uncertainty. Paired with split-by-reversibility: **two-axis trust model — confidence in data × reversibility of decision.** High confidence + reversible = fast path. Low confidence + irreversible = full governance. Maps to how engineering review boards think. Problem: complexity of two-axis trust.

### Combination 6
PCL as dumb bus | User always decides | Synara accumulates as rules | Workflows contain sessions | Everything governed including Claude | Claude assembles its own

**Feasible?** No. Fatal contradiction: user always decides + everything governed + PCL as dumb bus.

**Interesting:** Reveals that "PCL as dumb bus" is dead in any combination where governance matters. If the data layer has no opinion, governance has no leverage. PCL needs to be at least contract-aware. Also: Claude assembling context while everything including Claude is governed = recursive governance problem.

### Combination 7
Direct read/contract on write | Synara decides everything | PCL is system of record | Sessions = workflows (different views) | Everything governed including Claude | Lazy pull on demand

**Feasible?** With modification (Synara decides asynchronously — actions proceed, review within window).

**Interesting:** "Sessions and workflows are the same thing, different views" is a sleeper insight. One unified work object with two projections — user-facing and system-facing. **Eliminates an entire class of session-workflow boundary decisions.** Risk: single data model must satisfy both interactive and automated use cases. "Synara decides everything" = command economy, kills usability on the floor.

### Combination 8
PCL as dumb bus | Synara rules + Claude beyond | Claude reconstructs each session | Sessions graduate to workflows | Data contracts only | Synara assembles context

**Feasible?** No. "Claude reconstructs" + "Synara assembles" = two competing assemblers.

**Interesting:** The collision forces a valid split: Synara = data curation (decides what's relevant), Claude = sense-making (decides what it means). Separation between curation and interpretation. "Sessions graduate to workflows" is wasted because no persistent session to graduate if Claude reconstructs each time.

### Combination 9
Direct read/contract on write | Split by reversibility | Distributed knowledge | Sessions graduate to workflows | Everything governed including Claude | Lazy pull (with prefetch for governed decisions)

**Feasible?** With modification (mandatory prefetch for governed decisions).

**Interesting:** Claude's consequential recommendations are governed by reversibility. Claude recommending "look at this chart" = ungoverned. Claude recommending "reject this batch" = full Synara review. **Most practical AI governance: constrain consequential recommendations, not thinking.** Repeated patterns harden into workflow steps — Claude literally teaches the system new processes under governance supervision.

### Combination 10
Synara mediates everything | Synara rules + Claude beyond | PCL is system of record | Workflows contain sessions | Everything governed including Claude | Claude assembles its own

**Feasible?** No. Maximum Synara = god-object.

**Interesting:** Instructive failure. When Synara mediates access, makes decisions, AND governs, it becomes the exact anti-pattern SVEND avoids. **Lesson: Synara should own governance, not infrastructure.** The moment Synara mediates data access, it stops being governance and becomes middleware.

---

## Three Most Promising

### 1. "The Trust Architecture" (Combination 2)
Direct read/contract on write | Split by reversibility | PCL is system of record | Sessions graduate to workflows | Outcomes only governance | Claude assembles its own context

Every dimension reinforces the others. Fast where it can be, careful where it must be. Outcome-only governance avoids micromanagement. Sessions graduating captures natural maturation of exploratory work into standard process. Closest to how well-run manufacturing operations actually work.

### 2. "Tribal Knowledge Made Explicit" (Combination 4)
Direct read/contract on write | Split by reversibility | Synara accumulates as rules | Sessions graduate to workflows | Outcomes only governance | Session assembles context

Key difference from #1: knowledge lives in Synara's accumulated rules, not PCL's measurements. System intelligence grows through governance learning. Models how experienced operators carry process knowledge as internalized rules, not remembered data points. Session-assembles-context keeps Claude as pure facilitator.

### 3. "Governed AI Learning" (Combination 9)
Direct read/contract on write | Split by reversibility | Distributed knowledge | Sessions graduate to workflows | Everything governed including Claude | Lazy pull with prefetch

Most interesting from AI-governance perspective. Claude's consequential recommendations governed by reversibility, repeated patterns harden into workflows under supervision. Describes how the system learns safely over time.

---

## Unexpected Dimension Interactions

### 1. "Split by reversibility" + "Outcomes only governance" = emergent trust calibration
Reversibility governs WHEN to intervene; outcomes-only governs HOW to evaluate. Together: intervene at right moments, learn from right signals. Neither alone achieves this.

### 2. "Sessions graduate to workflows" is load-bearing
Appeared in all top 3. Not just a nice relationship — it's the mechanism for capturing institutional knowledge. Without graduation, sessions are disposable. Without sessions, workflows are rigid. The graduation path makes SVEND a learning system.

### 3. "Synara mediates everything" kills every combination it appears in
When Synara owns data path AND governance path = bottleneck god-object. Consistently better: direct read, contract on write. Anyone can look at the gauge; changing the setpoint requires sign-off.

### 4. "PCL as dumb bus" incompatible with meaningful governance
Data layer needs contract awareness. Eliminates 3,125 combinations (20% of space).

### 5. "Claude assembles" vs "Session assembles" is deeper than expected
Claude assembles = opinionated facilitator (decides what matters). Session assembles = auditable facilitation (UI/workflow decides what's relevant). First is more powerful; second is more auditable. For governed manufacturing: session-assembles may be safer default, Claude requests additional context when it identifies gaps.
