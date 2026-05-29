# Conference: SVEND Container System Design Spec

**Date:** 2026-05-04
**Technique:** S1/S2 Dialectical Debate
**Personas:** S1 — Product Manager (user value, speed, commercial viability) vs S2 — Code Quality & Compliance Manager (proven patterns, test coverage, SOC 2, audit trail integrity)

## S1 (Product Manager) Position

### 1. What's Really at Stake

This isn't an architecture decision. It's a survival decision. You have 1 paying user, 4 demos, and $4,500/mo MRR as the target. The question is: does spending 30 days on internal plumbing bring a single new customer closer to signing?

The spec is clean engineering but has no user story that ends with someone paying money. The 4/4 purchase triggers (tribal knowledge, audit reasoning, dollar quantification) require exactly one thing to demonstrate: a workflow trace that captures what someone did, why, and what it cost. The other 8 models are infrastructure for the second year, not the next demo.

### 2. Recommended Approach

Ship the workflow trace in week 1. Demo it in week 2. Build everything else only if demos validate.

**Week 1 (days 1-7): Workflow Trace — the minimum that captures tribal knowledge**
- WorkflowInstance and WorkflowStep only. No WorkflowDef yet (ad-hoc, FK nullable). No heuristics.
- Wrap `analysis/router.py` dispatch so every analysis run logs a WorkflowStep with inputs, outputs, tool, duration.
- Build one simple view: `/app/workflow/<id>/` — timeline of what was done, with what data, in what order, results. Printable. Auditable.
- Add `notes` field (reasoning chain), `cost_impact` decimal, `reasoning` text to WorkflowStep.

Three fields — notes, cost_impact, and outputs JSON — hit all three purchase triggers. No governance engine. No PCL. No Bayesian confidence. No plugin.

**Week 2 (days 8-14): Demo prep + one template**
- Build Cpk-to-PPAP as hardcoded guided workflow (not schema-driven engine).
- Connect to existing capability analysis. "Cpk was 1.12 → PPAP evidence package → estimated scrap reduction: $14,200/year."

**Week 3 (days 15-21): Validate or pivot**
- Run demos. Show workflow trace. Show dollar impact.
- If tribal knowledge resonates: add WorkflowDef for save/share.
- If audit trails resonate: add export/PDF and immutability.
- If something unexpected: pivot.

**Week 4 (days 22-30): Build what demos told you to build.**

### 3. Specific Changes

**Keep:** WorkflowInstance, WorkflowStep (simplified), router wrapping, Cpk-to-PPAP template (hardcoded).

**Cut from 30-day window:** WorkflowDef, Heuristic model, entire PCL app, GovernanceRule/Decision with Bayesian confidence, svend plugin, 6 of 7 templates.

**Modify:** Add notes, cost_impact, reasoning to WorkflowStep. Drop tool_version. Drop tier field.

### 4. Risks Acknowledged

- Technical debt from hardcoded template (3-5 day rewrite cost later)
- Fragmented data without PCL (Claude can use workflow context JSON)
- No governance-grade audit trail (workflow trace IS an audit trail — just not Bayesian-scored)
- Demos might fail anyway (cheaper to learn early)

### 5. What's Lost if Conservative Wins

- 2-3 weeks of demo-ready time
- Learning velocity (zero new market signal at day 30)
- Ability to pivot in week 3
- Energy and morale (solo founder building infrastructure nobody sees)

### 6. Steel-Man Against

"When a prospect says yes, you can't onboard them because you have no reusable definitions, no governance, no PCL. You close and scramble for 8 weeks." Response: The gap between demo and onboardable is ~2 weeks of focused work. Build for customer #2, not customer #50.

---

## S2 (Code Quality & Compliance Manager) Position

### 1. What's Really at Stake

This is open-heart surgery on a live compliance-critical system with no surgical team. Building 3 new apps + governance redesign + plugin in 4 weeks on the same server serving live users. The governance redesign touches `syn/` — the layer that enforces compliance on everything else. You're rebuilding the safety net while standing on it.

The real risk is compliance regression under time pressure. Week 3 arrives, integration is harder than expected, corners get cut on test coverage, immutability guarantees, migration safety, and compliance checks.

### 2. Recommended Approach

Split into two 4-week phases. Workflows + PCL in Phase 1. Governance redesign + plugin in Phase 2.

**Phase 1 (Weeks 1-4):** Scaffold apps, workflow engine + API, PCL models + API, control phase.
**Phase 2 (Weeks 5-8):** GovernanceRule/Decision, Contract model, wire governance to workflows, plugin, full regression + compliance audit.

Why: Governance is highest-risk. Building it under time pressure alongside two other apps guarantees shortcuts. Building after workflows/PCL are stable means testing against real executions, not mocks.

### 3. Specific Architecture Changes

1. **GovernanceDecision must be append-only with hash chain** — mirror SysLogEntry enforcement. Prevent UPDATE at Django level.
2. **WorkflowStep needs idempotency keys** — for crash recovery and workflow resumption.
3. **PCL confidence needs floor (0.01)** and override mechanism with CR linkage.
4. **Contract needs versioning** — `supersedes` FK for amendment history.
5. **Drop "learned" source from GovernanceRule** in Phase 1 — most complex, least auditable.
6. **Add MAX_WORKFLOW_DEPTH=3** — prevent recursive execution / runaway workflows.
7. **Feature flags per new app** — incremental activation, kill switch if issues.

### 4. Risks in This Approach

- 8 weeks delays market feedback
- Phase separation creates integration risk (mitigation: define governance interface contracts in Phase 1)
- Plugin ships late (mitigation: standalone pre-commit hook in Week 1)
- Psychological cost of slower pace for solo founder

### 5. What Goes Wrong if Ship-Fast Fails

- Governance mutability bug invisible until audit — can't retroactively prove immutability
- Confidence formula edge cases (observation_count=0 → NaN propagation)
- Custom workflow exposes engine limitation → patching live tables
- Compliance checks not updated for new relationship graph → false positives erode trust
- Partial migration failure on production database at 2 AM
- Meta-failure: weeks 5-8 spent fixing what shipped in weeks 1-4

### 6. Steel-Man Against

"You're optimizing for compliance posture that doesn't have external auditors yet. Prospects need capabilities, not audit trail perfection. Existing infrastructure provides the floor. Track record (22 extractions, 15 packages) says 4 weeks is credible." Counter: those were decomposition, this is composition. Composition under time pressure is where compliance gaps hide.

---

## Conference Synthesis

### Agreements

- **The full spec as written is too much for 4 weeks.** Neither endorses the original plan unchanged.
- **Workflow engine is the highest-value deliverable.** Both put WorkflowInstance and WorkflowStep in Week 1.
- **Governance redesign is the highest-risk component.** S1 cuts it entirely; S2 pushes to Phase 2. Neither wants it built under time pressure alongside other apps.
- **The plugin is low priority.** Both defer it.
- **PCL is not needed for the first demo-able artifact.** S1 cuts it; S2 includes in Phase 1 but acknowledges it could be deferred.
- **Hardcoded Cpk-to-PPAP is the right first template.**

### Disagreements

| Topic | S1 (Product Manager) | S2 (Compliance Manager) |
|-------|---------------------|------------------------|
| PCL | Cut entirely. Claude uses workflow context JSON. | Build in Phase 1. Typed measures enable querying and validation. |
| Timeline shape | 4 weeks total. Demo in week 2, pivot in week 3. | 8 weeks across two phases. Control phase at end of each. |
| "Done" at day 30 | Demo-able workflow trace + market signal collected. | Workflow engine + PCL + API + test coverage + compliance checks. No demos yet. |
| Governance | Cut. Workflow trace IS an audit trail. | Defer to Phase 2 but define interface contracts in Phase 1. |
| WorkflowDef | Cut. Hardcode first template. Customer-50 problem. | Keep. Needed to onboard customer #2 without bespoke work. |
| Test coverage | Existing infrastructure (SynaraEntity, CHG-001) is sufficient floor. | New apps require own compliance guarantees before production. |

### Crux of Each Disagreement

| Disagreement | Crux |
|---|---|
| **PCL** | Is unstructured JSON sufficient for Claude to reason about process state, or does Claude need typed, queryable measures? Empirical question about Claude's performance on real data. |
| **Timeline** | Is 30 days of plumbing without market signal more dangerous than shipping a thin demo that might expose limitations? Judgment about which failure mode costs more for a solo founder. |
| **"Done" at day 30** | Does the founder need market signal or architectural foundation more urgently? Is the bottleneck demand-side (nobody knows if people will buy) or supply-side (can't deliver what they want)? |
| **Governance interfaces** | Will workflows built without governance contracts require significant rework? Technical judgment about coupling. |
| **WorkflowDef** | How many customers can you onboard with hardcoded templates before the cost exceeds building it? S1 says many. S2 says fewer than you think. |
| **Test coverage** | Does existing compliance infrastructure provide adequate safety for new apps, or do new apps need their own guarantees before production? |

### Open Questions for the Arbiter

1. **What is the actual state of the 4 demos?** Are they waiting on workflow capability, or stalled for other reasons? If stalled on something else, S1's "demo in week 2" doesn't accelerate revenue.

2. **How much does Claude currently struggle without PCL?** If Claude loses context across sessions or confuses measures today, PCL has immediate value. If Claude functions adequately on current structures, PCL can wait.

3. **What happened during the 22 prior extractions?** Were there integration bugs? Does the track record support or contradict S2's claim that composition is harder than decomposition?

4. **Is the Ernie/Sikorsky demo a workflow demo?** If the most promising prospect needs workflow trace, S1's timeline is correct. If they need something else, the framing shifts.

5. **What is the realistic cost of a governance bug post-launch?** With no external auditors scheduled, is there a scenario where a governance gap costs a customer in the next 6 months?

6. **How much energy do you have?** S1 flags founder morale. S2's 8-week plan is methodical but doubles time before external validation. Which failure mode is more psychologically dangerous?

### Risk of Each Path

**If You Follow S1 (Ship Fast, Demo Week 2):**
- Demo exposes shallow implementation ("can I customize?" → "not yet")
- Onboarding gap after closed deal (~2 weeks scramble, possibly more)
- Unstructured workflow data becomes legacy to migrate
- No governance = no audit story for regulated prospects (Sikorsky)
- Technical debt accumulates under success pressure

**If You Follow S2 (Two Phases, 8 Weeks):**
- Zero market signal for 8 weeks — direction could be wrong
- Energy depletion — 8 weeks of infrastructure with no external validation
- 4 pending demos age out, warm prospects go cold
- Phase 2 integration still risky (deferred, not eliminated)
- Over-engineering for phantom requirements (no external auditors scheduled)
- Competitor or market shift during 8-week window
