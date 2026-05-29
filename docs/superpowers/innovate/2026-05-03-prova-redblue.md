# Red/Blue Analysis: PROVA

**Date:** 2026-05-03
**Technique:** Red Team / Blue Team
**Attack Dimensions:** Adoption/Usability, Technical Complexity/Build Cost, Competitive/Market Fit

---

## Blue Team Defense

### 1. The Problem and Why This Is the Right Solution

Manufacturing organizations generate enormous amounts of process data but lack a structured way to reason about it. The current state of the art is tribal knowledge in people's heads, scattered spreadsheets, and quality tools (SPC, DOE, FMEA) used in isolation. When a quality engineer suspects that ambient humidity affects coating adhesion, that hypothesis lives in a conversation, an email, or nowhere at all. When they run a test, the results live in a report that gets filed. The causal understanding -- "we tested this, here's what we learned, here's how it changes what we believe about our process" -- is never captured as a first-class object.

This is not an abstract problem. It has concrete costs: duplicate investigations (team B re-discovers what team A learned two years ago), contradictory beliefs operating simultaneously on the floor, and inability to propagate the implications of new evidence across related processes.

PROVA solves this by making process beliefs explicit, structured, and evidence-linked. The key insight is that manufacturing causal knowledge is not free-form text -- it is a graph of IF-THEN relationships between measurable quantities. Temperature affects viscosity. Viscosity affects flow rate. Flow rate affects coating thickness. These are edges with directionality and measurable endpoints. By constraining hypotheses to structured graph edits, PROVA forces clarity without requiring statistical sophistication from the user.

The two-layer architecture (Operating/Working) mirrors how knowledge actually works in manufacturing. There is what we know (validated through evidence) and what we suspect (floor observations, hunches, correlations we noticed). Mixing these is dangerous -- it leads to acting on unvalidated beliefs. Separating them completely is impractical -- it kills the pipeline from observation to knowledge. The two-layer design keeps both visible while makes the boundary explicit.

### 2. What Makes This Design Robust

**Structured hypotheses eliminate ambiguity.** Free-text hypotheses ("I think temperature matters") are untestable. A graph edit -- "add edge: mold temperature -> part shrinkage, hypothesized direction positive, proposed trial: 10 shots at 3 temperature levels" -- is testable by construction. The form-based builder is not a limitation; it is the discipline that makes the system work.

**Computed evidence weight removes gaming.** Letting users assign evidence strength is an invitation to confirmation bias. Computing discriminating power (how well does this evidence distinguish between competing graph versions?) is objective and automatically rewards experiments that actually resolve disagreements. A user cannot inflate the importance of evidence that supports their preferred theory.

**Conflicts break links rather than auto-resolve.** Auto-resolution algorithms would hide the most valuable information the system can surface: "your process model is contradicted by evidence." By breaking the link and surfacing the economic cost of the contradiction, PROVA forces human judgment where it belongs -- on the resolution -- while the system handles what it's good at: tracking and propagating.

**Bottom-up construction from PCL.** Users already maintain process characteristics (temperatures, pressures, cycle times) in their value stream maps. PROVA edges emerge naturally: "we measured these two things, they seem related, let's formalize that." This avoids the cold-start problem of asking users to build an abstract graph from scratch.

**Damped propagation with physical boundaries.** Using physical cycle boundaries (job length, batch size) as damping points is grounded in reality -- the process literally resets at these points -- rather than being an arbitrary algorithmic parameter that would need tuning.

### 3. Anticipated Failure Modes and Mitigations

- **Cold start / empty graph:** Mitigated by building bottom-up from PCL.
- **User resistance to structure:** Mitigated by green/blue/purple complexity tiers.
- **Confirmation bias:** Mitigated by path-of-least-resistance design and two-question filter.
- **Over-engineering trials:** Mitigated by validating simple 10-row datasets.
- **Stale graphs:** Mitigated by active propagation signals.

### 4. Key Assumptions

- Manufacturing causal knowledge can be represented as a directed graph of measurable quantities (supported by Ishikawa, fault trees, FMEA).
- Users will invest in structured hypothesis entry (because the alternative produces nothing actionable).
- Small trials are sufficient (supported by Taguchi philosophy).
- A solo developer can build this (foundation libraries already exist, Django layer is integration).

### 5. Competitive Advantage

No existing manufacturing SaaS captures causal process knowledge as a first-class, evidence-linked, version-controlled graph. Minitab, JMP, and SAS do statistical analysis -- they do not maintain a living model of how your process works or propagate the implications of new evidence across that model. The integration advantage is decisive: SVEND already has SPC, DOE, RCA, FMEA, Ishikawa, VSM, Control Charts, and Capability Studies as evidence sources. The practical-significance-first philosophy directly serves the target user: the quality engineer who needs to improve a process, not publish a paper.

---

## Red Team Attacks

*Ranked by likelihood x impact.*

### 1. Nobody Is Asking For This (HIGH x CRITICAL)

No market signal supports building PROVA now. 1 paying user, 4 demos, 60 outreach emails -- no prospect has asked for a causal knowledge graph. Quality managers justify purchases with "this replaces Excel SPC" or "this automates FMEA documentation." Nobody walks into a budget meeting saying "I need a Bayesian causal reasoning engine." The bottom-up PCL approach requires users to already be engaged with VSM/PCL -- which are themselves unproven market-fit features.

### 2. Dependency Stack Trap (HIGH x CRITICAL)

Five v0.1.0 libraries (forgesia, forgestat, forgedoe, PCL, Synara) must all be correct, compatible, and stable before PROVA delivers value. PCL is still in design. A bug in forgesia means PROVA gives wrong answers. The design has already pivoted three times (DSW deprecated, analysis workbench migration, now PCL-first). Each pivot resets the clock. Worst case: 18 months from now, PROVA is 80% built across five libraries, none production-hardened, revenue still under target.

### 3. Cold Start Death Spiral (HIGH x HIGH)

Operating Graph starts empty, grows only through completed trials. New customers must: populate PCL, build initial edges, design and execute trials, wait for evidence to accumulate. Weeks or months of investment before value exceeds a whiteboard. Manufacturing environments are high-pressure, low-patience. If the tool doesn't show value in the first session, the champion loses internal credibility.

### 4. Vocabulary Wall (HIGH x HIGH)

Target users think in Ishikawa diagrams, 5-Why chains, and A3 templates. PROVA asks them to think in "graph edits," "competing graph versions," "premise truth frequencies." A quality engineer who suspects temperature affects viscosity doesn't think "challenge the edge between nodes 47 and 112." The green/blue/purple tiers address complexity but not vocabulary translation.

### 5. Opportunity Cost (HIGH x HIGH)

Every PROVA month delays: tutorial/onboarding that converts trial users, Gemba Exchange SEO, enterprise readiness features (SLA, audit export, IP allowlist), and polishing existing SPC/DOE/FMEA tools that are actual purchase triggers.

### 6. Propagation Is a Research Problem (MEDIUM x HIGH)

Real manufacturing has nested feedback loops with variable time constants (thermal drift across shifts, tool wear across weeks, seasonal material variation across months). Getting propagation right is not a well-solved library problem -- it's the kind of thing academic papers get written about.

### 7. Operator Contribution Is Fantasy (MEDIUM x MEDIUM)

Operators contribute messy, verbal, contextual observations. The Working Graph requires structured outcomes with committed trial dates. The two-question filter kills most operator contributions before they enter the system, eliminating ground-truth signal.

---

## Adjudication

### Covered by Defense

| Finding | Verdict | Notes |
|---------|---------|-------|
| Propagation complexity (#6) | PARTIALLY COVERED | Physical cycle boundaries are a reasonable first-order answer. Only a research problem if you need it optimal. "Signal that related things changed" is straightforward for MVP. |
| Cold start (#3) | PARTIALLY COVERED | PCL bottom-up approach helps graph population but doesn't address the value-delay problem. Solvable with seed templates, industry-standard edges, or FMEA import -- but the defense doesn't propose any of these. |

### Genuine Vulnerabilities

| Finding | Decision Relevance | Why It Matters |
|---------|-------------------|----------------|
| Nobody is asking for this (#1) | HIGH | The blue defense argues the problem exists but never connects that to market demand. "No existing SaaS does this" could just as easily mean nobody wants it. This dictates timeline, not viability. |
| Dependency stack trap (#2) | HIGH | The blue defense doesn't engage with the v0.1.0 reality. PCL-first strategy actually concentrates all risk on the least-proven component. This is the failure mode that consumes 18 months invisibly. |
| Vocabulary wall (#4) | MEDIUM | UX problem, not architecture problem. Solvable by mapping PROVA operations to familiar quality tool language (Ishikawa arrow = edge, 5-Why chain = path, A3 = hypothesis lifecycle). But needs explicit design work. |
| Opportunity cost (#5) | HIGH | Doesn't require PROVA to be wrong -- only requires other work to be more valuable right now. With 1 paying user, every PROVA month is a month not spent on conversion infrastructure. Sequencing argument, not viability argument. |
| Operator contribution (#7) | LOW-MEDIUM | Operators are not the primary user. But if the design claims this as a feature, it needs a real pathway for unstructured observations. |

### Red Team Missed

1. **PCL depends on Synara, making the chain six layers deep.** The dependency stack is actually deeper than the red team identified -- the foundation of the foundation (Synara) is itself recently migrated.
2. **Graph version control under concurrency.** The blue defense mentions "version-controlled graph" as competitive advantage but the design doesn't specify how versioning interacts with concurrent users, branch-and-merge of competing hypotheses, or rollback when evidence is retracted.

### Most Important Finding

**The opportunity cost attack (#5) is the most decision-relevant, but the dependency stack trap (#2) is the most dangerous.**

Opportunity cost changes what you do Monday morning. The dependency stack is the failure mode that consumes 18 months invisibly -- each library seems close to done, the integration never quite works, and by the time you recognize the trap you've sunk a year.

**The honest summary:** PROVA is a sound design for a real problem, but the blue defense fails to justify building it *now*, fails to address the dependency stack risk concretely, and fails to account for the vocabulary gap between the design's abstractions and the user's mental model. The bottom-up PCL pivot is architecturally correct but concentrates risk on the least-mature component. None of these are fatal to the concept. All of them argue for sequencing it after the revenue foundation is proven.
