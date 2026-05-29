# Focus Group: Integration Design in SVEND

**Date:** 2026-05-03
**Technique:** Research-Grounded Focus Group
**Prompt:** What does well-designed integration look like in a system like SVEND? What works? What's overkill? Where is there value versus imposition of unwelcome structure?

## Personas

- **Greg Linden** — Quality Director, 340-person tier-2 automotive stamping (Michigan). Integration Survivor. Catastrophic SAP QM rollout left him allergic to platform pitches.
- **Priya Chakraborty** — Solo Quality/Compliance Manager, 95-person food ingredients plant (New Jersey). Spreadsheet Drowning Victim. 14 Excel workbooks, no IT, desperate for integration but terrified of overengineering.
- **David Kwon** — Manufacturing IT Manager, 600-person medical device company (Minneapolis). Integration Plumber. Maintains 11 point-to-point integrations between 7 systems. 30% of his time = keeping plumbing running.
- **Marcus Wade** — CNC machinist / 2nd-shift quality lead, 45-person job shop (Cincinnati). Shop Floor Pragmatist. Killed an InfinityQS trial because it added 5 minutes per check. Zero purchasing authority, absolute veto power.

## Research Summary

Personas grounded in: Elsmar Cove forums (EQMS/ERP integration), CIO/Panorama ERP failure case studies, Capterra/G2 reviews (Minitab, ETQ Reliance, MasterControl), Practical Machinist forums (shop floor data collection), Manufacturing Dive (spreadsheet costs), food safety compliance surveys, Struto/HappierIT integration cost analyses, MachinMetrics/Tulip operator studies, Zaptic Industry 4.0 failure data. Key pattern: >70% of Industry 4.0 investments fail to move beyond pilot; operators fail to enter data 50% of the time when the system adds friction.

---

## Persona Responses

### Greg Linden (Integration Survivor)

**First reaction:** "You just listed ten things... That's not a product pitch, that's a consulting deck." Heard the same slide in the SAP presentation in 2018. "Knowledge graph" = meaningless jargon he doesn't have time to care about. Has a 7am scrap meeting to worry about.

**What would make him care:** ONE problem solved. CMM data from PC-DMIS currently goes through Excel, Minitab, PPAP workbook — 15x/week during launch season, 20-30 min each, copy-paste creates transcription errors that become Ford customer quality notices. "Drop your CMM file here, get your Cpk here, export the PPAP page here." Also needs auto-built scrap Pareto and better CAPA tracking — but won't buy it all at once. "I tried buying everything at once. It was called SAP."

**What would stop him:** $15K approval authority; >$30/user/month for inspectors is DOA. Must run side-by-side with Minitab — won't rip out muscle memory on a promise. Burned three times (SAP QM, InfinityQS, consultant's "close the loop" integration that created mandatory fields nobody fills). 58-year-old lead inspector needs to learn it in a morning.

**What he'd need to see:** Actual PC-DMIS export file → Cpk study in Ford STA format. Inspector does it in 2-minute walkthrough. Scrap data → auto Pareto by 7am. Phone number of a quality manager at a 200-500 person IATF shop, 6+ months live. 30-day trial, one use case, his data.

**How he'd pitch it:** "There's a tool that cuts CMM-to-PPAP data handling from thirty minutes to five and eliminates copy-paste errors — I want to trial it on the F-150 bracket launch." Full platform pitch? "I'd nod politely, take your card, and never call you."

### Priya Chakraborty (Spreadsheet Drowning Victim)

**First reaction:** "That's not a product description, that's a conference brochure... built for a consulting firm to sell to a Fortune 500 company, not for me." One person with 14 Excel workbooks and an SQF audit in September. DOE, FMEA, VSM, knowledge graph = built for companies 10x her size.

**What would make her care:** Customer complaint on lot 2024-0847 → CAPA → root cause → verification → training → supplier COA in 2 minutes instead of 45. "Call it a knowledge graph, call it traceability, call it whatever. Just make it work without me spending 3 months configuring it." Also: document control replacing her manual Excel register. Doesn't need SPC (pass/fail), DOE (never run one), FMEA (Word doc from 2021), or VSM. Key question: "Can I ignore the 80% I don't need without it getting in my way?"

**What would stop her:** Price >$12K/year. Implementation fees doubling Year 1. "90 days to go-live" = maintaining Excel AND learning new system at 55-hour weeks. Vendor lock-in (watched previous employer get trapped in MasterControl). If logging an NCR takes more clicks than an Excel row, it's failed. Hidden feature tiers where real needs are in "Enterprise." And: "If I have to 'request a demo' to find out what it costs, I'm already annoyed."

**What she'd need to see:** 15-minute live demo: create NCR → link CAPA → assign action → attach document → record training → pull full chain from search bar. Reference: food/pharma <200 employees, SQF/BRC, no IT person, live in <60 days. 30-day free trial. Pricing on the website.

**How she'd pitch it:** "It's a cloud system that links our CAPAs to complaints and documents automatically — it'll cut my audit prep from three weeks to three days, and it costs less than what we'd pay a temp during audit season." Full pitch? "I'd tell him it's built for companies ten times our size."

### David Kwon (Integration Plumber)

**First reaction:** "Who is this for?" Person doing SPC isn't doing DOE isn't writing FMEAs isn't managing CAPAs — five different roles in one platform. "Knowledge graph" is either genuinely interesting architecture or marketing for "we have foreign keys in our database." Has heard this pitch as "digital thread" (PTC), "closed-loop quality" (every QMS vendor). Reality: replace three tools, add two integration points, spend $25K on validation, floor supervisors still export to Excel.

**What would make him care:** The traceability gap BETWEEN systems. Cpk drops below 1.33 in Minitab → nothing happens automatically → someone manually creates CAPA in MasterControl → references Minitab file by name in a text field → FDA auditor asks to see the chain → quality director manually reconstructs across three systems and a shared drive. "If your knowledge graph means 'when SPC detects a shift, automatically create a structured corrective action with data already linked' — okay, now I'm listening. If it means 'pretty visualization showing connections' — I already have Visio." BUT: has to be BETTER at SPC than Minitab or he's adding system #8.

**What would stop him:** Validation cost ($15-30K IQ/OQ/PQ per new system). API stability — needs versioned endpoints, 12-month deprecation policy, no breaking changes on minor releases (Epicor broke lot traceability with a "minor update," 40 hours to fix, 6-week CAPA). Data model opacity — "Can I see your schema? Not your API docs — your actual data model." "Replace everything" pitch = 2-year project that gets deprioritized. Team of 2 can't spare 200 hours for implementation.

**What he'd need to see:** What happens with ugly data — nulls from skipped LabVIEW fields, leading zeros stripped from lot numbers, 8-point capability studies, false positive SPC signals, mid-sync connection loss. "Show me your system handling ugly real-world manufacturing data." Reference: ISO 13485 medical device, similar size, small IT team. 60-90 day sandbox with real data and full export.

**How he'd pitch it:** Probably wouldn't — not yet. IF traceability gap solved with real data: "There's a tool that automatically links process measurements to corrective actions, so when FDA asks how we detected a problem and what we did about it, the answer is one click instead of four hours."

### Marcus Wade (Shop Floor Pragmatist)

**First reaction:** "You just said nine things. I need maybe one and a half." Doesn't know what a knowledge graph is. "Integration of what? I have a CMM, a paper chart, and a pen. Those are already integrated. I measure, I write, I look, I decide. Takes 90 seconds." Gut: "this sounds like a system for the quality manager and the CI director that I'll be asked to feed data into."

**What would make him care:** CMM auto-capture → live chart at his machine that updates when PC-DMIS finishes → trend flag before he'd catch it on paper. "The second you make me open a browser, remember a password, select a part number from a dropdown, or enter dimensions by hand, you've already lost." Also: his NCRs go into a folder and he never knows what happens. Same failure mode shows up again two months later.

**What would stop him:** Zero purchasing authority (can only say no). Trust deficit — every demo is clean hands and a mouse; reality = coolant on screen, wifi drops, barcode scanner won't read. Paper is fast, reliable, doesn't crash. No IT support at 7pm. Any data entry = DOA.

**What he'd need to see:** A 40-60 person job shop where this runs ON THE FLOOR, not in the quality office. Talk to operators, not quality manager. CMM connection working live — actually pull from Brown & Sharpe running PC-DMIS, chart updates without keyboard. 30-day trial on one machine.

**How he'd pitch it:** IF auto-capture works: "Hey, there's a system that pulls our CMM data automatically and gives me a live SPC chart at the machine. Catches trends faster than paper." If any data entry required: "I'd just quietly stop using it."

---

## Synthesis

### Consensus (4/4 convergence)

**"Solve one problem first, not ten."** All four rejected the breadth of the feature list. Greg: "consulting deck." Priya: "conference brochure." David: "who is this for?" Marcus: "I need maybe one and a half." A ten-capability pitch triggers pattern-matching against failed enterprise implementations, not excitement about comprehensiveness.

**Trial before commitment, on my data, with my mess.** All four demanded 30-60 day trials using actual data — not demo data, not clean data. Nobody offered to buy sight-unseen. The trial is the purchasing mechanism.

**Reference customers matched to my size and industry.** Every persona specified tight match criteria. Generic case studies satisfy none of them.

**"Integration" means traceability between events, not feature count.** When personas described valued integration: CMM data to Cpk to PPAP page, complaint to CAPA to verification, SPC signal to corrective action, NCR to resolution and back to the operator. Nobody described integration as "all tools in one platform." They described it as "when X happens, Y follows without me re-entering data."

**Learning curve as hard veto.** 58-year-old inspector needs to learn it in a morning. NCR must take fewer clicks than Excel. Browser login and dropdown already too much. Floor supervisors will export to Excel regardless. Ease of adoption is survival, not preference.

### Divergence

**What "integration" means splits by role:**
- Greg/Marcus (hands-on): data flows from instrument to document without re-keying. Physical-world integration.
- Priya (solo compliance): events chain together so she can pull one thread for an auditor. Document-world integration.
- David (IT): systems talk via APIs with structured handoffs. Architecture-world integration.

**Appetite for analytical tools diverges completely.** Marcus: SPC only if zero-entry at his machine. Greg: Cpk for PPAP. David: "at least not worse than Minitab." Priya: doesn't need SPC at all. DOE relevant to nobody in this group.

**Trust deficit varies by specific scar.** Greg: SAP catastrophe. Priya: MasterControl lock-in. David: Epicor API breakage. Marcus: InfinityQS trial. Generic reassurance misses all four.

### Strongest Objection

**Greg: "I tried buying everything at once. It was called SAP."**

Most damaging because Greg IS the ideal customer — quality director, mid-size manufacturer, real pain, real budget, real purchasing frequency. His objection is not about price, features, or competition. It is about the shape of the pitch itself. A comprehensive platform pitch triggers his SAP pattern-match, and once triggered: polite disengagement. This reaction is likely representative of a large segment — quality directors at 200-500 person manufacturers have almost all survived at least one failed enterprise implementation.

### Unmet Needs Discovered

**The NCR black hole (Marcus).** "My NCRs go into a folder and I never know what happens. Same failure mode shows up again two months later." The operator is the detection system and gets no signal back. Not a feature request — a reason he doesn't trust systems.

**Audit prep as the real time sink (Priya).** Her value proposition is not "better quality" — it is "cut audit prep from three weeks to three days." A purchasing trigger the prompt didn't anticipate.

**Validation cost as the hidden price (David).** IQ/OQ/PQ costs $15-30K regardless of software price. For regulated industries, the software price may be the smaller number.

**Offline resilience (Marcus).** No IT support at 7pm. System goes down = back to paper = running two systems. Any system that can't degrade gracefully will be abandoned by the people it most needs to serve.

### Would Pay

| Persona | Would buy? | Entry point | Timeline |
|---------|-----------|-------------|----------|
| Greg | Yes, one capability at a time | CMM-to-PPAP on F-150 launch | 30-day trial → purchase |
| Priya | Yes, if price visible and go-live fast | Complaint-to-CAPA chain | Weekend evaluation → 60-day go-live |
| David | Not yet — sandbox first | Traceability gap between SPC and CAPA | 60-90 day sandbox → 6-12 month decision |
| Marcus | Cannot buy, can only veto | CMM auto-capture at his machine | Adoption or quiet abandonment by day 14 |

### Language Mining

The market's words, not ours:

- **"Copy-paste errors" / "transcription errors"** — the failure mode, not "lack of integration"
- **"Customer quality notices"** — the consequence that creates urgency
- **"Pull the full chain"** — Priya's mental model of traceability
- **"The traceability gap BETWEEN systems"** — David's precise diagnosis
- **"Manually reconstructs the chain"** — what FDA audits actually require today
- **"Data I'll be asked to feed into"** — Marcus's mental model of any new system
- **"Live chart at my machine"** — Marcus's definition of value
- **"Quietly stop using it"** — how systems actually fail (silence, not complaints)
- **"Side-by-side with Minitab"** — coexistence, not replacement
- **"Mandatory fields nobody fills"** — the specific mechanism of previous integration failure
- **"Can I ignore the 80% I don't need without it getting in my way?"** — the small-company platform question
- **"Foreign keys in our database"** — David's cynical translation of "knowledge graph"
- **"Request a demo to find cost"** — a disqualifying friction point
- **"Coolant on screen, wifi drops"** — the physical reality demos never show
- **"I measure, I write, I look, I decide"** — the workflow any tool must beat, in eight words
- **"Audit prep from three weeks to three days"** — Priya's purchase trigger, fully formed
- **"One click instead of four hours"** — David's internal pitch, if it works
