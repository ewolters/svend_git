# Conference: SVEND Separation of Concerns

**Date:** 2026-05-04
**Technique:** S1/S2 Dialectical Debate
**Input:** Morphological analysis (15,625 combinations → 3 candidates)

---

## S1 (Innovator) Position

### Recommends: Option A — The Trust Architecture (with "Trust but Verify at the Seam")

**Framing:** This is a bet on where intelligence lives. Option C is fear architecture — governing Claude's recommendations is like making your best engineer fill out a form before thinking out loud. Option B creates two competing knowledge stores that will diverge within 6 months.

**Architecture:**
- PCL is the ONLY knowledge store. Synara does NOT accumulate knowledge. Clean separation: PCL = what we know, Synara = what's allowed.
- Claude pulls from PCL directly. No intermediary context assembly. Claude follows reasoning chains you didn't predict.
- Reversibility-based judgment mirrors manufacturing: adjust a parameter (reversible) vs scrap a lot (irreversible).
- Enhancement: `context_cited` field on writes — snapshot of PCL entries Claude referenced. Auditor reconstructs not just WHAT changed but WHY Claude recommended it. More powerful than governing the recommendation itself.
- Session graduation is event-driven: session emits Synara event, workflow consumes (pull-only).
- Outcomes feed back into PCL via Bayesian update. System learns without a separate learning engine.

**Risks acknowledged:**
- Claude hallucinates context: mitigated by `context_cited` populated from actual PCL queries, not Claude's summary.
- No governance on recommendations: ungoverned suggestions with governed actions is SAFER — operator knows they own the decision.
- 11 days tight: each piece independently shippable.

**What's lost if conservative wins:** Speed (governance latency in 90-second window), emergent insight (capped by governance rules), simplicity (three new subsystems), the learning loop (two things to update), positioning ("Claude guides" vs "Claude guides after our system checks Claude").

**Steel-man for Option C:** Regulated environments may need recommendation governance. Auditor asks "why did the system recommend that?" and Option A logs evidence but not approval gate. Response: build A, add governed-recommendation mode as enterprise toggle when enterprise customer demands it.

---

## S2 (Conservative) Position

### Also Recommends: Option A — The Trust Architecture (phased)

**Framing:** Where does truth live when things go wrong at 2 AM? Three concrete questions: Can you trace a recommendation in 90 seconds? Is institutional knowledge auditable and independent of Claude's session memory? When a recommendation costs $50K in scrap, who signed off?

**Architecture:**
- PCL stores knowledge. "Tool 7 drifts high after 200 cycles on aluminum" = PCL characteristic with provenance, not Synara rule.
- Synara governs, does not know. Contracts, audit trail, approvals. Clean scope.
- Claude pulls context at query time. No prefetch engine. No context cache. Simple pull.
- Reversibility gate is binary. Reversible or irreversible. No gray "medium risk."
- **Sessions do NOT graduate to workflows in Week 1.** Phase 2, after observing what operators actually do.

**Phased delivery:** Foundation weeks 1-2, iterate based on reality weeks 3-4. Add graduation IF operators create repeatable patterns. Add prefetch IF pulls are too slow. Add recommendation governance IF outcomes-only proves insufficient.

**Risks acknowledged:**
- PCL bottleneck: flexible characteristics + metadata handle most knowledge types. Watch for friction.
- Claude context assembly may be slow at scale: Bayesian confidence provides natural filtering. Measure before prefetching.
- Outcomes-only governance may miss dangerous reasoning: periodic review, sample-based audit — exactly what quality systems recommend.
- Less "impressive" in demos than governed AI.

**Steel-man against own position:** PCL stores characteristics but NOT relationships between them. Relational knowledge ("when Tool 7 drifts AND humidity > 60% AND third shift, check coolant") needs structure that survives context windows. Response: start with A, add lightweight rules layer inside PCL (not Synara) if Claude consistently misses relational knowledge.

---

## Conference Synthesis

### Agreements

Both agents converge with no tension on:

- **Option A wins.** Neither advocates for B or C.
- **PCL is single system of record.** Dual knowledge stores will diverge.
- **Claude assembles its own context.** Session-assembled = new middleware to maintain. You'll get it wrong.
- **Reversibility-based judgment.** Binary gate, no gray area.
- **Outcomes-only governance.** Don't govern thinking, govern acting.
- **Option C rejected.** Governance overhead kills 90-second window; 11 days insufficient for distributed governance.
- **Option B rejected.** Knowledge in Synara rules diverges from PCL.
- **Enterprise governance toggle deferred.** Future add for regulated customers.

Unusually high agreement. Substantive disagreements are narrow but consequential.

### Disagreements

| Topic | S1 (Innovator) | S2 (Conservative) |
|---|---|---|
| Session graduation timing | Ships in 11-day build. Event-driven, independently shippable. | Phase 2. Observe real usage before designing graduation triggers. |
| `context_cited` audit trail | Per-write snapshot of PCL entries Claude referenced. Day 1 requirement. | Not mentioned. Relies on periodic review and sample-based audit. |
| Relational knowledge | Implicit: Claude's reasoning over multiple PCL entries handles it. | Explicit concern: PCL stores characteristics, not relationships. Claude may miss multi-factor interactions. |

### Crux of Each Disagreement

**Session graduation:** Does shipping graduation logic before observing real usage create wasted work or capture an opportunity? S1: event-driven pattern is simple, won't need rework. S2: wrong triggers are worse than no triggers.

**`context_cited`:** Is per-write provenance necessary from Day 1, or is sampling sufficient? Deeper question: will a regulated customer (Sikorsky, aerospace) require per-recommendation traceability?

**Relational knowledge:** Can Claude reliably infer cross-characteristic relationships from flat PCL reads? S1 bets yes. S2 bets it fails on complex conditional relationships. Question is empirical — but cost of discovering late is operator trust erosion during trust-formation window.

### Open Questions for the Arbiter

1. How many sessions will operators run before graduation matters? Dozens = S2 is right (defer). Multi-session problem-solving in week one = S1 is right (ship it).
2. Is Sikorsky likely to require per-recommendation traceability? AS9100 typically requires objective evidence of decision rationale.
3. Do you have concrete examples of multi-factor relational knowledge that Claude needs across sessions? Test them against flat PCL reads now.
4. What is the actual 11-day scope? S1 includes graduation. S2 excludes it. Different "done."
5. Is the first paying user or ILSSI audience more likely to need graduation or relational knowledge handling?

### Risk of Each Path

**S1 path (full build):**
- Graduation logic encodes wrong triggers, requires rework (medium severity, medium-high likelihood)
- 11 days tight with graduation + context_cited; something ships half-baked (medium)
- Relational knowledge gap unaddressed (high severity, unknown likelihood)

**S2 path (foundation, defer graduation):**
- Operators develop manual workarounds, habits calcify (medium)
- No per-write audit trail, regulated customer asks for retrofit (medium)
- "Add rules layer later" becomes permanent debt (medium)
- Ships less, learns less (low severity, high likelihood)

**Shared risk:**
- PCL as sole system of record bottlenecks if schema isn't rich enough (high, unknown)
- Claude context assembly degrades as PCL grows to thousands of entries (medium, 6-12 months)
- Outcomes-only governance insufficient for regulated customer, requiring Option C bolt-on under pressure (high severity, low near-term)
