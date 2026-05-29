# Focus Group: Forge Ecosystem Coverage — Does the 21-Engine System Cover Prospective Customers?

**Date:** 2026-05-12
**Technique:** Research-Grounded Focus Group
**Personas:**
- Greg Fontaine — Quality Director, contract packaging (Cincinnati), Champion
- Keisha Aldridge — Reliability Engineering Manager, chemical plant (Houston), Skeptic
- Linh Nguyen — QA Manager, snack manufacturer (Portland), Adjacent
- Dale Whitaker — Aerospace job shop owner (Wichita), Constraint

## Research Summary

Sourced from Elsmar Cove forums (SPC software evaluation, reliability software recommendations, AS9100 burden for small shops), Practical Machinist (SPC in machine shops, free SPC tools, ISO/AS9100 threads), Quality-One PPAP documentation, SQFI Quality Code Level 3 requirements, SafetyChain/NWA food industry SPC marketing, Pardus Consulting free Weibull template, ReliaSoft pricing, QC-Calc/InfinityQS pricing, ECI Solutions AS9100 cost analysis, Capterra Minitab alternatives. Four segments identified: multi-customer contract manufacturer (report reformatting pain), reliability engineer with zero software budget (Excel-based Weibull), food QA manager pursuing certification (non-statistician needing guided workflow), aerospace job shop owner (short-run SPC, ITAR, offline).

## Persona Profiles

### Greg Fontaine — Quality Director, TriStar Flexible Packaging
- 200 employees, Cincinnati, 14 CPG customers simultaneously
- 3 quality engineers maintaining separate Excel workbooks per customer
- 40% of engineering time spent reformatting — same analysis, different customer PPAP formats
- 5 Minitab licenses ($8,925/yr) + InfinityQS real-time SPC ($18K/yr) = $27K/yr
- Quarterly lean events, GM tracks waste on P&L
- Burned by InfinityQS, ETQ, Sparta Systems — all delivered QMS, not statistics
- Signing authority up to $15K without VP approval

### Keisha Aldridge — Reliability Engineering Manager
- 350-employee food-grade chemical blending plant, Houston, OSHA PSM
- 120 rotating equipment assets, 8 years of Maximo CMMS data
- Weibull analysis in free Excel template (Pardus Consulting)
- Tried to get ReliaSoft ($5K/seat) — told to "make do"
- Knows pump #1 is Weibull beta 2.3 (wear-out), can't get capital team to act
- Burned by "predictive AI" vendor (needed sensors) and RCM consultant (no tools)
- Zero software budget

### Linh Nguyen — QA Manager, Pacific Coast Snacks
- $60M revenue, Portland, SQF Level 2 pursuing Level 3, 180 production employees
- SafetyChain ($24K/yr) for food safety, limited SPC (X-bar/R, p-charts)
- Not a statistician — SQF auditor told her she needs "statistical evidence of quality system effectiveness"
- Previous Minitab purchase: engineer left, nobody else could use it, $8K wasted
- Knows 30+ QA managers from SQFI conferences (network multiplier)

### Dale Whitaker — Owner, Whitaker Precision Machining
- 22 employees, Wichita, AS9100D certified, 8 aerospace primes
- 70% of work is lot sizes 5-50 (low-volume, high-mix)
- QC-Calc ($3,200/yr) + JobBOSS ($4,800/yr) + AS9100 audits + calibration service
- One quality person (Kevin) who also does receiving inspection
- AS9100 auditor flagged MSA program last year
- Spotty WiFi, no IT staff, ITAR work, subscription fatigue

## Persona Responses

### Greg Fontaine

**First reaction:** "Twenty-one engines — that's a marketing number. I don't care about 21. I care about three, maybe four, that actually work correctly." The flowchart concept made him lean forward — one data source branching into fourteen different report templates maps his actual problem. "ETQ said 'report builder.' What they meant was a PDF export with their logo on it."

**What would make him care:** "Solve the reformatting problem. That's the $27K question." Engineers spend 40% of time copying numbers between customer-specific Excel workbooks. If the platform lets him define customer output templates — P&G format, Purina format — and the same underlying data populates all of them, "I will write you a check this afternoon." Gage R&R with real ANOVA tables dropping into customer-formatted reports.

**What would stop him:** (1) Plant floor usability — operators in gloves on tablets need big buttons, not flowchart canvases. Two different UX needs: office engineers vs floor operators. (2) 7 years of InfinityQS data migration. (3) Startup trust — "$49/mo is startup pricing." (4) Whether it replaces InfinityQS for real-time SPC or only replaces Minitab.

**What he'd need to see:** Live Gage R&R on his data, output in P&G's PPAP format. Phone call (not case study PDF) with another contract manufacturer.

**Pitch to boss:** "Could eliminate the reformatting work eating 40% of our engineers' time and replace both Minitab and potentially InfinityQS at less than half the combined cost."

---

### Keisha Aldridge

**First reaction:** "This is a platform looking for a problem." But "Weibull analysis with both MRR and MLE" made her stop — "That's specific. That tells me someone who actually does reliability work touched this." AMSAA-Crow for measuring whether PM restructuring bent the failure intensity curve. System reliability for serial pump/heat-exchanger trains.

**What would make her care:** Weibull end-to-end with censored data. AMSAA-Crow with before/after segmentation. The Weibull-to-FMEA closed loop (CDF at mission time → occurrence rating automatically). DES for maintenance crew staffing. "Four of them to work really, really well."

**What would stop her:** (1) "Twenty-one engines means you're spreading development across twenty-one things. If they exist at the expense of the reliability module being half-baked, I'd rather you had five engines that were bulletproof." (2) "When I see '320 statistical tests' I think who validated all of those?" (3) Painful Maximo CSV import. (4) Reports that look like grad student homework.

**What she'd need to see:** Weibull matching Abernethy/ReliaSoft within 1%. AMSAA-Crow with confidence bounds. Weibull-to-FMEA closed loop on realistic data. DES maintenance scenario in under 30 minutes. Cost clarity — no upsell walls.

**Pitch to boss:** "It does the Weibull I've been doing in Excel plus the system reliability modeling we need a ReliaSoft license for. $49/mo instead of $5,000/seat. I can build the capital justification for the #1 pump replacement."

**Bottom line:** "The question is whether 'Weibull MLE' means you implemented scipy.stats.weibull_min.fit and called it a day."

---

### Linh Nguyen

**First reaction:** "Twenty-one engines? Flowcharts? Typed ports? You lost me in the first sentence." She needs one question answered — is her fill weight capable? — and one document proving it. "You have way more analytical power than I need, and that's actually the problem."

**What would make her care:** "Say these exact words: 'Upload your checkweigher CSV. We'll tell you if your process is capable and generate the report your SQF auditor needs for Level 3 clause 2.5.2.'" Template mode is what she needs but it was buried under power-user language.

**What would stop her:** (1) Jargon terror — 21 options on first screen = closed tab. (2) "The Kevin problem" — if she leaves, can QA tech Maria run it? (3) No food/SQF context — needs "SQF Level 3 fill weight capability report" not "capability study template."

**What she'd need to see:** 3-minute video: CSV upload → audit-ready report. The actual report shown to her SQF consultant. Proof guided mode is step 1/2/3/done. One food manufacturing reference.

**Pitch to boss (current):** "Built for engineers, not QA people." **Pitch (fixed):** "Does the capability studies for SQF Level 3 for $49/mo. Maria could run it."

**Bottom line:** "Bury the 21 engines. Show me three words: SQF Level 3."

---

### Dale Whitaker

**First reaction:** "Twenty-one things that can break during a customer audit." Internet dependency during audits is existential. "I run a 22-person job shop. I'm not Toyota. You built a Swiss Army knife when I need a torque wrench."

**What would make him care:** (1) AS9102 FAIR output (Forms 1, 2, 3) — Kevin spends 6 hrs/week on FAIRs. (2) Short-run capability for lots of 12-30 without warnings that scare customers. (3) Gage R&R with guided setup that passes audit.

**What would stop him:** (1) Internet dependency — "If your servers are down during an audit, I can't tell Textron 'come back Thursday.'" (2) ITAR — server location, access controls, SOC 2, NIST 800-171. (3) 8 years of QC-Calc/Excel migration. (4) Subscription pricing — $49 for two users fine, team tier for two logins = walk away.

**What he'd need to see:** Ten measurements → capability report → something he could hand to a customer, under 5 minutes, live. Gage R&R walkthrough. What happens with WiFi off. FAIR template.

**Pitch to Kevin:** "Try it on that 7075 bracket for Spirit. Same data in both tools, see if numbers match. If reports are clean and we don't need IT, we'll talk. If internet goes down and we lose data, it's dead to me."

**Bottom line:** "Don't sell me the platform. Sell me six hours a week of Kevin's time back."

## Synthesis

### Consensus

**"I need four things that work perfectly, not twenty-one that exist."** All four independently rejected breadth as a value proposition. Greg: "three, maybe four." Keisha: "four of them to work really, really well." Linh: "way more power than I need — that's actually the problem." Dale: "Swiss Army knife when I need a torque wrench." The 21-engine count actively works against the product in every segment.

**Customer-formatted output is the actual product.** Greg's 40% reformatting waste. Dale's 6 hrs/week on FAIRs. Linh's SQF auditor document. Keisha's capital justification report. Every persona independently identified "report that satisfies an external party" as the deliverable they're paying for — not the computation.

**Trust and continuity risk.** Greg: Series A. Dale: uptime during audits. Keisha: validation rigor. Linh: bus factor. Same question, four frames: can I depend on this when stakes are real?

**Data migration is a gate.** Greg: 7 years InfinityQS. Keisha: Maximo exports. Dale: 8 years QC-Calc/Excel. No evaluation without a credible migration answer.

**Gage R&R appeared in 3 of 4 responses.** Most cross-cutting analysis type.

### Divergence

**Computational depth vs. guided simplicity.** Keisha wants MLE confidence bounds and NIST verification. Linh wants step 1, step 2, step 3, done. Not a spectrum — two fundamentally different experiences sharing an engine. Template mode vs build mode is the right architectural answer but must be executed as truly separate experiences, not a toggle.

**Internet dependency.** Dealbreaker for Dale (ITAR + audit scenarios), non-issue for the other three. Aerospace-specific but absolute within that segment.

**Price signal direction.** Greg and Keisha see $49/mo as suspiciously cheap (triggers trust concerns). Dale sees it as fine if two users are included. Linh sees value but won't buy what her team can't use.

**Incumbent replacement target.** Greg: InfinityQS + Minitab. Keisha: ReliaSoft. Dale: QC-Calc. Linh: SafetyChain. Competitive positioning must be segment-specific.

### Strongest Objection

Keisha: **"Twenty-one engines means you're spreading development across twenty-one things. If they exist at the expense of the reliability module being half-baked, I'd rather you had five engines that were bulletproof."**

Structural, not situational. Dale's internet concern is solvable. Greg's trust concern fades with traction. Linh's jargon problem is UX. But Keisha questions whether the architecture itself guarantees mediocrity — whether building 21 engines makes it impossible to be excellent at any one. She's the persona most capable of verifying technically. If Weibull confidence bounds don't match ReliaSoft, the entire "comprehensive platform" narrative collapses.

### Unmet Needs Discovered

**Customer-specific output templating** — Greg's reformatting problem. Same underlying data rendered into P&G format, Purina format, Unilever format. The document engine exists but customer-specific form templates are not addressed.

**Industry-standard form generation** — Dale needs AS9102 FAIR Forms 1/2/3. Linh needs SQF Level 3 clause 2.5.2 reports. Specific regulated forms with specific field mappings, not generic documents.

**Short-run SPC methods** — Dale's lot sizes of 5-50 break standard Cpk assumptions. DNOM charts, Q-charts, pre-control — not listed in the SPC engine.

**Organizational resilience (bus factor)** — Linh's "Kevin problem." Can a less-technical person operate this when the primary user leaves? A purchase criterion, not a nice-to-have.

### Would Pay

- **Greg — YES.** Has budget, acute pain, articulated pitch. Needs live Gage R&R in customer format + reference call.
- **Keisha — YES, conditionally.** No budget but self-funding pitch (capital justification tool). Must match ReliaSoft within 1%.
- **Dale — MAYBE.** Would pay $49/mo for two users. Internet dependency and ITAR are structural blockers.
- **Linh — NO, not today.** Not price or capability — accessibility. Would buy "SQF Level 3 report tool" instantly. Won't buy "21-engine platform."

### Language Mining

Phrases the personas used that the prompt did not:
- "Reformatting problem" — the prompt says "document generation"; the customer says "reformatting"
- "PPAP format" / "FAIR Forms 1, 2, 3" / "SQF Level 3 clause 2.5.2" — specific deliverable names that are the actual unit of value
- "Six hours of Kevin's time back" — ROI as recovered person-hours
- "Operators in gloves on tablets" — physical ergonomic context
- "Bent the failure curve" — reliability engineer's outcome language
- "Closing the tab" — the actual behavior when jargon overwhelms
- "Report builder — what they meant was a PDF export with their logo on it" — vendor scar tissue
- "Startup pricing" — $49 as signal of impermanence, not affordability
- "Swiss Army knife when I need a torque wrench" — breadth as liability
- "Platform looking for a problem" — technology-first framing detected
- "Hit by a bus" — organizational continuity as purchase criterion
- "Come back Thursday" — audit scenario where downtime has consequences
- "scipy.stats.weibull_min.fit and called it a day" — the specific shortcut she's screening for
- "Numbers match" — validation = identical output to incumbent, not better output
