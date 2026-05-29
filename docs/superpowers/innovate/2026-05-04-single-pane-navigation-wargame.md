# War Game: Single-Pane Navigation Without Claude

**Date:** 2026-05-04
**Technique:** Multi-Turn Simulation
**Roles:** Marcus (Floor Operator), Lisa (Quality Engineer), New User (Free Tier, No Claude), Marketplace (Catalog/Analytics)
**Turns:** 4 (7 tools → 15 → 30 → 55)

## Game Design

**Scenario:** SVEND's single-pane UI must be navigable WITHOUT Claude. Everything — SPC, capability, CAPA, VSM, Hoshin, calculators, marketplace add-ins — lives in one view. How is it organized? What happens as the marketplace grows from 7 to 55+ tools?

**Roles:**
1. **Marcus (Floor Operator)** — CNC operator. SPC only. 90 seconds. Tablet with coolant. Will abandon if >2 taps. Private: considering paper if slow.
2. **Lisa (Quality Engineer)** — 11yr Minitab. Capability, SPC, CAPA, Pareto, Gage R&R, PPAP daily. Private: will organize by "apps" mentally regardless of UI.
3. **New User (Free Tier, No Claude)** — Just signed up. CSV data, wants capability study. 30 minutes or abandon. Private: comparing to Minitab Express.
4. **Marketplace** — Not a person. Growing catalog. Private: 80% of add-ins used by <5% of users.

---

## Turn-by-Turn Log

### Turn 1 — Launch, 7 Tools

**[Marcus]:** Opened SPC Control Charts on tablet. Couldn't find Part #4471 in 90-second window. Abandoned. Asked Lisa to set it up.

**[Lisa]:** Explored all 7 tools, mentally categorized (statistical, problem-solving, process, utility). Configured SPC for Part #4471, pinned it. Set up Marcus's tablet with SPC pinned as default. Opened Capability + CSV Import side by side.

**[New User #1]:** Found CSV Import by scanning for "Upload." Imported CSV. Opened Capability Analysis as second panel. Stalled ~8 minutes trying to connect imported data to the Capability tool — composition model opaque without Claude.

**[Marketplace]:** SPC = anchor tool (opened first by 2/3 users). CSV Import → Capability = primary sequence. Pareto, Gage R&R, CAPA = Day 1 orphans. Flagged import-to-analysis handoff as highest-leverage UX fix. Flagged Lisa's proxy-setup pattern as primary deployment model for operators.

### Turn 2 — 3 Months, 15 Tools

**[Marcus]:** Pinned SPC loaded automatically. 12 seconds in-app. Sidebar bigger (15 items + "Recent" section), ignored entirely. Asked Lisa "they changed something — everything still work?"

**[Lisa]:** "Recent" section validated her mental model. Opened PPAP Assembly (she requested it), tested Hypothesis Testing (worked). Noticed no way to hide/reorder sidebar. Pinned PPAP. Told Marcus "yours is fine." Did not mention new tools to Marcus.

**[New User #2]:** Scanned 15-tool alphabetical sidebar, found CSV Import and Capability. Same handoff friction — 22 min to first Cpk. 8 new tools were invisible noise.

**[Marketplace]:** SPC 89%, Capability 74%, CSV Import 68% = Tier 1. 6/8 new tools below 5%. Recommended: contextual suggestions, Getting Started bundle, hiding tools, absorbing CSV Import. Key finding: adding tools makes product worse for everyone except the requester. Flat sidebar won't survive 30 tools.

### Turn 3 — 9 Months, 30 Tools

UX improvements shipped: Favorites section, categories (Statistical, Problem-Solving, Process, Quality System, Utilities), data bus (shared data picker across tools), contextual suggestions after analysis, search box.

**[Marcus]:** Used search to find OEE Calculator (supervisor mandate). Found it, but default view showed analytics, not data entry. Found Downtime Log sub-tab. Logged event. 3:40 total. Starred OEE.

**[Lisa]:** Searched FMEA Builder — loved data bus (SPC data auto-populated in FMEA). Tested Regression (adequate, not starred). **Became gatekeeper:** approved gage study tool (clear methodology), trialed report builder (no preview), rejected "Smart SPC" (dynamic control limits = statistical malpractice). Sent team rejection note.

**[New User #3]:** Data bus eliminated handoff friction. 4 min to first Cpk (down from 22). Contextual suggestion → Histogram. Starred both. 11-min session. Did not discover what makes SVEND different from Minitab Express.

**[Marketplace]:** Data bus reduced new-user time 14.2→3.8 min. Favorites = primary nav (68% of multi-session users, avg 7.2 favorites). Search = primary discovery (41%). Categories used only first 2 sessions, then abandoned. 17/30 tools below 5%. Third-party: 6 tools, avg 3.1/5 rating vs 4.2 first-party. One complaint (Smart SPC). Recommended: Verified Methodology badge, absorb CSV Import, role-based defaults. Key insight: users build own 5-tool micro-products. Sidebar is for onboarding, not daily use.

### Turn 4 — 18 Months, 55 Tools

UX improvements shipped: Role-based onboarding, CSV Import absorbed into data bus, Verified Methodology badge (7/20 third-party), tool bundles (e.g., "New Product Launch"), quick-action bar, workspace templates, Site Admin permission.

**[Marcus]:** Nelson Rule 2 violation triggered "Initiate CA" button on SPC chart. Tapped it. CA form required description, root cause, severity, assignment. Couldn't fill assignment field (unknown person directory). Supervisor helped. 4:20 total. Told supervisor "this used to take me a minute."

**[Lisa]:** Installed NPI bundle (FMEA + PPAP + Control Plan + Capability) — loved it, data bus tight, recommended to team. Tried Bayesian Capability — appreciated posterior distribution viz. Filed formal complaint against PdM Pro (no confidence intervals, no validation). Posted warning in site channel. Heard Marcus's CA complaint, will raise UX issue. Spent 95 min on tool evaluation/governance, pushing actual QE work back.

**[New User #4 — Lean/CI Coordinator]:** Selected "Lean/CI" role. Got pre-configured favorites (Gemba Walk, Hoshin, Skills Matrix, Poka-Yoke, 5S Audit). Gemba Walk worked — 6 min to value. Hoshin felt thin — no data bus integration, no live KPIs, a structured spreadsheet. Searched for VSM, A3, kaizen — no/wrong results. 22-min session. Starred Gemba only. Did not return within 7 days.

**[Marketplace]:** Operator retention 97% (CA friction). QE retention 94%. Lean/CI 30-day retention 11%. NPI bundle 34% adoption among QEs. Data bus: 31/35 first-party connected, 4/20 third-party, 1/5 Lean. Search 52%. Categories deprecated. 17 tools below 5% (long tail). Recommended: Lean tools need data bus or fail, operator CA is UX anti-pattern, governance must go from volunteer to institutional, free tier is Lean dead end, reframe from "55 tools" to "5 workflows, shared data, AI-guided."

---

## Private Reasoning (Revealed)

### Marcus — Across All Turns

- **Turn 1:** Gave it one honest attempt. Gap between "open SPC tool" and "see my chart" was too wide. Needs a bookmark/destination, not a tool. Will never try the launcher again.
- **Turn 2:** Brief anxiety at sidebar change, but pinned chart loaded. Mental model fully reduced to "the app that shows my SPC chart." Unaware 14 other tools exist. **This is a success state.**
- **Turn 3:** Search saved him. Without it, 30 tools = impossible. OEE default view showed analytics (Lisa's model), not data entry (his model). Different information architectures for different users inside the same tool.
- **Turn 4:** 90-second equilibrium broken by CA form. Assumes quality-engineer mental model for operator interaction. Marcus is a sensor, not a workflow initiator. **Will start ignoring Nelson rules or dismissing alerts to avoid paperwork.** Satisfaction dropped.

### Lisa — Across All Turns

- **Turn 1:** Treating it like Minitab menu. Pin/default-view for Marcus is critical. Considers setting up Marcus part of her job.
- **Turn 2:** Mentally dividing tools into "mine" (7) and "not mine yet." Wants "my tools in my order," not just recent. Did not tell Marcus about new tools — would create anxiety.
- **Turn 3:** Most important development: made domain-expert judgment marketplace cannot make. Rejected technically functional but methodologically wrong tool. No structured evaluation framework. FMEA + data bus = retention feature. Regression not starred because "correct output" is table stakes.
- **Turn 4:** At inflection point. 95 min on evaluation/governance. Actual QE work pushed back. Wants: (1) formal tool approval workflow, (2) delegation to other reviewers, (3) recognition this work matters. **Curation burden unsustainable.**

### New Users — Across All Turns

- **Turn 1 (#1):** Two-step import-then-analyze requires 3 conceptual steps vs Minitab's 1. Composition model opaque without Claude. No middle ground between figuring it out and bouncing.
- **Turn 2 (#2):** Alphabetical ordering accidentally helped. Import-to-analysis handoff still highest friction. 8 new tools were noise.
- **Turn 3 (#3):** Radically better (data bus). But performed Minitab Express workflow — SVEND hasn't shown differentiation. Free tier without Claude = "decent online calculator." May not return.
- **Turn 4 (#4, Lean/CI):** Exposed structural gap. Lean tools are second-class — standalone forms competing against spreadsheets. Will tell management "quality stats tool with Lean bolted on." **Will advocate against standardization on SVEND.**

### Marketplace — Across All Turns

- **Turn 1:** Import-to-analysis handoff = highest-leverage fix. Lisa's proxy setup = primary operator deployment pattern.
- **Turn 2:** Adding tools makes product worse for everyone except requester. Marketplace paradox.
- **Turn 3:** Three tensions: growth vs coherence (users build own 5-tool products), third-party quality (Smart SPC = canary), free-tier differentiation (moat invisible or paywalled).
- **Turn 4:** Three divergences: retained excellent / new-user stuck at 26%; integrated tools retain / island tools churn; governance at critical juncture. **SVEND succeeds as QE platform and operator console but fails as "single-pane manufacturing platform." Lean/CI is the proof.**

---

## Narrator Analysis

### 1. Narrative Summary

The game tested whether SVEND's single-pane navigation could function without Claude as the marketplace grew. What emerged was not a navigation problem but a **fragmentation problem disguised as a growth problem.**

Turn 1 established the foundational dynamic: Marcus outsourced to Lisa, the new user hit the composition wall, and the product's actual information architecture began being defined by Lisa, not by the sidebar. Turn 2 locked the pattern — Marcus reduced SVEND to one screen, Lisa began curating for others. Turn 3 was the pivot: the data bus solved the structural UX problem while Lisa's Smart SPC rejection proved the platform could not solve governance through infrastructure. Turn 4 broke equilibrium: the CA form un-solved Marcus's 12-second workflow, Lisa's curation became unsustainable, and the Lean/CI coordinator falsified the "single-pane manufacturing platform" identity.

The surprise: the product worked extremely well for a narrow corridor — SPC-anchored quality engineering with a competent Lisa — and every attempt to widen that corridor degraded the experience for people already inside it.

### 2. Emergent Findings

**The Lisa Bottleneck is load-bearing.** Lisa simultaneously became deployer (configuring Marcus), governor (rejecting tools), trainer (reassuring Marcus), and feedback channel (relaying complaints). These four functions converged on one person because the platform provided no formal surface for any of them. The platform's retention numbers measure Lisa's effort, not the product's design.

**Marcus and new users solve different problems that look identical.** Marcus needs a destination (single screen, his data). New users need a path (sequence to first value). The sidebar serves neither. The data bus fixed the path but not the destination. The CA form violated the destination by inserting a path into it. These are architecturally incompatible needs sharing one navigation surface.

**The marketplace has analytical clarity but no execution authority.** Every turn it produced the correct diagnosis. Every turn the diagnosis arrived too late or lacked an execution mechanism. It functions as a historian, not a governor.

**Tool addition is negative-sum past a threshold.** Each tool adds value for its requester and friction for everyone else. The NPI bundle was the first positive-sum addition because it was pre-composed — four tools as one workflow. **The atomic unit of value is the workflow, not the tool. The catalog's unit of inventory is the tool. This mismatch is structural.**

### 3. Vulnerabilities Surfaced

1. **Governance vacuum.** Lisa is the entire governance layer. No formal approval workflow, no delegation, no recognition. When she leaves, governance drops to zero instantly with no graceful degradation.

2. **"Single-pane" identity falsified.** Lean/CI coordinator proved SVEND's depth is domain-specific (quality/statistics), not universal. Every Lean user who arrives expecting a manufacturing platform and finds a statistics platform leaves with a damaging narrative.

3. **CA form broke the operator contract.** Marcus's 90-second equilibrium was the product's best outcome. The CA button violated it by assuming operators are workflow participants rather than sensors. His private "will start ignoring Nelson rules" is alert fatigue — safety-relevant degradation.

4. **New-user conversion stuck at ~25%.** The moat (data bus, AI facilitation, cross-tool integration) is either invisible infrastructure or paywalled behind Claude. Free tier = "decent online calculator."

5. **Third-party quality is a trust multiplier.** One quality escape from a bad tool damages the entire platform, not just that tool. Lisa's interception is the only defense. The platform has no second line.

### 4. Information Asymmetry Insights

- Lisa and Marcus have a **stable, protective information asymmetry** the platform keeps trying to collapse. Lisa deliberately withholds tool info from Marcus. The CA form collapsed this by injecting QE concepts into his interface.
- Marcus's "will ignore Nelson rules" is a **safety signal no one can see.** Lisa will hear it as UX complaint, not detection-system degradation.
- New users' post-session narratives ("decent calculator," "Lean bolted on") **propagate through organizations** in channels the platform cannot monitor.
- Lisa's Turn 4 frustration reads as **high engagement in analytics** (95 min session, deep feature exploration). The platform cannot see that curation time is displacing engineering time.

### 5. Key Decision Points

**Primary inflection: Turn 3, Lisa rejects Smart SPC.** Before this, every problem was solvable through engineering (data bus, favorites, search). Lisa's rejection introduced a problem requiring domain-expert judgment the platform cannot replicate. The platform has three options: formalize Lisa's role (scales her burden), build algorithmic quality gates (cannot replicate her judgment), or ignore it (leads to trust damage). The simulation showed option 3 by default.

**Secondary inflection: Turn 4, CA button on Marcus's SPC chart.** The moment the solved problem (Marcus's 12-second equilibrium) was un-solved by a well-intentioned feature.

### 6. Unspent Threats

**Claude AI was the largest unspent force.** Deliberately withheld from the simulation. Its shadow shaped everything: the data bus is Claude's substrate, the free-tier "calculator" perception exists because Claude is the withheld differentiator, Lisa's curation burden exists because Claude could theoretically provide recommendations. The simulation tested structural integrity without the primary load-bearing feature. **Result: without Claude, the product converges to "Lisa + favorites + search" as its navigation system.**

**Lisa's ability to leave was never deployed.** Her departure is the highest-impact unspent threat. Operator retention, quality governance, deployment, and organizational trust all depend on her continued voluntary participation. Every metric the platform reports is conditionally valid: valid if Lisa stays.

**No competitor appeared.** The Lean/CI coordinator's exit narrative ("quality stats with Lean bolted on") is the seed of a competitive positioning SVEND's own user generated.

**The quality escape never happened.** Lisa intercepted both problematic tools. Survivorship bias in real-time. The platform appears safe because Lisa's unpaid labor prevented the event. No second line of defense exists. **The platform does not know Lisa is the defense.**

**Marcus never trained a second operator.** Every new operator is a deployment task for Lisa. Scales linearly with headcount.
