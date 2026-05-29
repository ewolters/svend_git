# Focus Group: What Professionals Hate About Quality/OpEx Software

**Date:** 2026-05-03
**Technique:** Research-Grounded Focus Group
**Personas:**
- Dana Kowalski — CNC machinist / in-process inspector, 120-person auto stamping (Daily Use)
- Marcus Chen — IT Manager, 400-person med device mfg (Implementation/Admin)
- Rachel Torres — VP Operations, 200-person aero/defense contract mfg (Cost/Value/Lock-in)
- James Okafor — Sr. QE / Black Belt, 800-person semiconductor packaging (Power User Ceiling)

## Prompt Tested

"What do manufacturing and quality professionals hate about quality/OpEx/SPC/statistical software they've actually used? Real complaints about buying, implementation, daily use, reporting, pricing, vendor relationships, and what happens when they try to leave."

## Persona Profiles

### Dana Kowalski — Daily Use Torture
CNC machinist / in-process inspector at 120-person Tier 2 auto stamping plant, Grand Rapids MI. 14 years on the floor. No college degree. Uses InfinityQS ProFicient and WinSPC daily because quality engineering set it up. Previously used QI Analyst. Every 50 parts: stop press, walk 40 feet to shared terminal (one for three operators), wait if occupied, log in (45 seconds to "genuinely forever"), navigate to part/operation, enter 5 measurements, acknowledge alarms, walk back. 4-7 minutes. Paper log takes 30 seconds.

Sources: Practical Machinist forums, Elsmar Quality Forum, SourceForge reviews, InfinityQS 2021 survey (55% of Tier 1/2 auto suppliers still do capability in Excel), Ease.io LPA research.

### Marcus Chen — Implementation/Admin Hell
IT Manager at 400-person med device manufacturer. CS degree. 8 years. Responsible for MasterControl QMS, ERP integration, 21 CFR Part 11 validation. Didn't choose MasterControl — inherited it. 30% of time on MasterControl admin. Every update triggers re-validation (IQ/OQ/PQ), 3-6 weeks each, 4-6x/year. 15+ help desk tickets/week. Evaluated ETQ ($100-200K implementation) and Arena (rigid workflows). Annual cost $40-60K not including labor.

Sources: G2, Capterra, Software Advice, Propel competitive teardowns, CogniDox analysis.

### Rachel Torres — Cost/Value/Lock-in
VP Operations at 200-person contract manufacturer (aero/defense). MBA. $85K combined annual software spend: Minitab ($1,500+/seat, now subscription-only), Greenlight Guru (~$40K/yr), Plex quality module (bundled). 12 Minitab licenses but 40 engineers who need occasional stats. Minitab's float-to-named-user change cut coverage from 40 to 12. CEO asks "what's the ROI?" quarterly.

Sources: Trustpilot, TrustRadius, Capterra, G2, CogniDox, Atonement Licensing, CloudNuro.

### James Okafor — Power User Ceiling
Sr. Quality Engineer / Black Belt at 800-person semiconductor packaging. MS Industrial Engineering. 11 years. Runs SPC program, writes control plans, trains Green Belts. Uses Minitab daily, hits walls weekly. Needs batch capability across 200+ parameters, automated reporting, custom control chart rules, SPC-MES integration. Taught himself R and Python; company won't approve. Previously used JMP (no multilevel models), SAS ($50K+/seat).

Sources: G2, Capterra, Knowledge Academy, JMP Community, Medium, LinkedIn, Elsmar.

---

## Persona Responses

### Dana Kowalski

**The rant:** "I hate that I was hired to run a progressive die press and somehow half my job turned into being a data entry clerk on a computer from 2014." 4-7 minutes per data entry cycle vs. 30 seconds on paper. Shared terminals — if Ricky's on it, she waits. InfinityQS freezes during login. Can't correct her own typos (admin has to search the database). Acknowledged 400 alarms last year, maybe 10 resulted in anyone coming to her press. The incident: press running hot, walked to terminal to flag trend, terminal occupied, waited, software froze, 200 parts ran unaccounted, quality engineering blamed her.

**What she works around:** Paper log taped to press guard. Enters data at break (batch entry, timestamps wrong). Everyone does this. Quality engineering knows. Keeps own reject count on sticky notes — scrap module inaccessible. "The system thinks I entered them at 10:15 AM. I actually measured them at 9:22, 9:31, 9:40, 9:48, and 9:57."

**What nobody asks:** What happens between measurement and data entry (where all risk lives). How long the full loop takes (nobody's timed it). What she does when the system is down (runs parts with no digital record). Whether she understands control charts (she does — 14 years — but system treats her as data entry terminal). Whether she'd prefer something different (decision made by people who never run a press).

**What would help:** Data entry AT the press. See her own trend right there. Fix her own mistakes with reason code + audit trail. Alarms that actually do something (text QE, create ticket, escalate). Speed above all.

**To the vendor:** "You built your software for the quality manager's office, not the shop floor. Come stand at my press for one shift. Just one. Then tell me your software is working as intended."

### Marcus Chen

**The rant:** "MasterControl sells itself as a quality management system. It's not. It's a document management system with a CAPA workflow bolted on." Every object is an "InfoCard" — same layout, menus, metadata whether CAPA, document, or training record. 15+ help desk tickets/week for 6 years (~4,500 tickets) because UI provides zero context. Validation lifecycle: every vendor update triggers re-validation, quarter of his year spent on it. "Validation on Demand" saves nothing. Forced migration from Classic on-prem to cloud: 11 months, $180K, lost three custom reports that took 2 years to build. SOAP for half the API, REST for the other half, different auth. 14-month-old support ticket unresolved. Reporting requires third-party tools that bypass permissions model.

**What he works around:** Shadow SharePoint with linked Excel workbooks mapping MasterControl → PLM → ERP. Not validated, not compliant, only way to answer auditor questions in <48 hours. Crystal Reports hitting database directly (built-in reports can't filter custom fields). Nightly Python script reconciling training completions between MasterControl and HRIS (scraping their API). Two terminated employees showed "trained" in QMS — "system functions as designed."

**What nobody asks:** What it actually costs (license is 40% — 600 hrs/yr admin labor = $80-90K). Recovery plan if vendor changes (2,000 hours to re-index exported PDFs). What he'd build if not working around the QMS.

**What would help:** Real data model (CAPA ≠ document ≠ training record). Real integration story (RESTful API, webhooks, OAuth, cross-system audit trail). Vendor owns validation burden. Consumption-based pricing.

**To the vendor:** "I'm not your customer because I chose you. I'm your customer because leaving you is more expensive than staying. That's not loyalty. That's a hostage situation."

### Rachel Torres

**The rant:** Minitab killed floating licenses → named-user: coverage from 40 engineers to 12, same spend. User's perpetual license broke after Windows Update — Minitab said "buy a new subscription." Greenlight Guru auto-renewed at 7% increase after team decided to migrate. Missed cancellation window by ~2 weeks. Owed $43K for system being actively replaced. No structured data export — contractor spent 3 weeks pulling documents. Paying for two QMS simultaneously. CEO asks "what's the ROI?" — no vendor helps build an ROI case that survives a CFO. "The vendor lock-in isn't a side effect. It's the business model."

**What she works around:** Bought 5 QI Macros at $329/each (perpetual, Excel) for 8 Minitab seats. "Nobody complained." Manual contract-trap-avoidance spreadsheet. Controlled Excel templates for Plex workflows.

**What nobody asks:** Total cost of ownership beyond license. How many people actually use it. What it costs to leave on day one.

**What would help:** Usage-based pricing. 30-day out contracts. Structured data export button. Price transparency. "The vendor I'd actually trust would say: 'Here's what it costs. Here's what it does. Here's how to leave. Here's your data if you do.' Four sentences. Nobody says them."

**To the vendors:** To Minitab: "You had forty loyal users. You priced thirty-six out. I replaced you with a $200 Excel add-in and nobody complained. You've lost me as an advocate permanently." To Greenlight: "You charged me $43K for a system I told you I was leaving. I've told this story twenty times. You earned a reputation."

### James Okafor

**The rant:** "Minitab is a teaching tool that we've all collectively agreed to pretend is an industrial tool." 47-parameter characterization, 3 equipment sets. Minitab: 141 manual analyses, 6 dialog boxes each, 3 days clicking. R script: 4 hours. Manager: "but Minitab is our approved tool." No scripting API. No automation. No pipeline. Cpk assumes normality — semiconductor distributions are Weibull, bimodal, beta. Gets "Johnson transformation and a prayer." JMP can't do multilevel models or generalized linear mixed effects. InfinityQS: getting data out is its own project. SAS does everything, costs $50K+/seat. IT won't approve R — validation of open-source "too complex."

**What he works around:** Dual workflow — official through Minitab (for audits), real in R/Python on personal laptop. R scripts: monthly capability across 200+ parameters in 10 minutes vs. 2 full days in Minitab (24 days/year). Custom SPC rules for semiconductor failure modes exist only in R scripts. Personal Shiny dashboard for real-time process characterization during NPIs. "This is what validation theater costs. Not just my time. Decision latency."

**What nobody asks:** What the analytical workflow actually looks like (needs a pipeline, not point-and-click). What he CAN'T do (vendor demos features he figured out 5 years ago). Cost of manual analysis. What control chart rules he actually needs (6 years of validated rules on his laptop). "Analyses not performed" — process relationships never investigated because tool makes exploration impractical.

**What would help:** Real programming interface: `run_capability(data, spec_limits, method="bayesian")`. Native mixed effects, Bayesian capability, non-normal without forced transformations. Batch/pipeline architecture. MES/ERP integration. User-definable control chart rules. Analysis and documentation as one artifact.

**To Minitab:** "Your competitor isn't other statistics packages. It's a Black Belt with a laptop, four hours, and R. The only reason I'm clicking your dialog boxes is IT won't approve the tool that already replaced you on my personal machine."

---

## Synthesis

### Consensus (4/4 independent agreement)

1. **The tool serves the buyer, not the user.** Dana: "customer is the guy who signs the PO." Rachel: paying for capability nobody uses. James: dual workflow because approved tool isn't the real tool. Marcus: 4,500 help desk tickets. Designed for the purchasing moment, not the working moment.

2. **Shadow systems are universal and known.** Dana: paper logs. Marcus: SharePoint + Python. James: R on personal laptop. Rachel: QI Macros + Excel. Every persona built parallel systems. Management is aware. The official system is compliance theater; the shadow system is the operation.

3. **Exit cost is the business model.** Rachel: $43K for a system she's leaving. Marcus: "hostage situation." James: audit trail locks him in. Dana: no influence on the decision. Retention strategy is switching cost, not satisfaction.

4. **Pricing penalizes breadth of use.** Rachel lost 28 Minitab users in licensing change. Marcus pays 2.5x license in admin labor. James: license + 24 days labor/year. Dana: invisible cost of minutes × thousands of cycles.

5. **Reporting and data access require workarounds.** Marcus: Crystal Reports hitting DB directly. James: translates between R and Minitab. Rachel: no structured export from Greenlight. Dana: can't see her own scrap data.

### Divergence

**Speed vs. power vs. control vs. integration.** Four fundamentally different jobs-to-be-done:
- Dana: seconds, not minutes. Anything beyond "enter five numbers, see chart" is friction.
- James: depth and programmability. A computational environment, not a GUI.
- Rachel: commercial control. Pricing, portability, contract terms.
- Marcus: architectural sanity. Real data models, real APIs, manageable validation.

Not reconcilable into one interface. Explains why monolithic quality platforms fail everyone.

### Strongest Objection

**James's "analyses not performed" gap.** Every other complaint is visible pain (slow entry, bad pricing, trapped data). James identified invisible cost: analyses that never happen because the tool makes exploratory work impractical. Process relationships never investigated. Early warning signals never detected. Shows up as a yield excursion six months later, nobody connects the two events. If the tool fails the power user this completely, it's a compliance artifact, not an engineering tool.

### Unmet Needs Discovered

1. **Measurement-to-decision latency** — Nobody has timed the full physical-digital loop. The risk surface exists between the process and the digital record.
2. **Validation burden as hidden tax** — Every vendor update triggers re-validation. Regulated customers are penalized for vendor improvements. Incentive to stay on old versions.
3. **Contract terms as product feature** — Rachel tracks renewal windows in a spreadsheet. Commercial relationship requires its own tooling to manage.
4. **Power user needs environment, not application** — James doesn't want better dialog boxes. He wants analysis, documentation, and deployment as one artifact in code.

### Anti-Patterns (named, for checklist use)

| # | Anti-Pattern | Description |
|---|-------------|-------------|
| 1 | **Buyer-User Split** | Designing for PO signer, not 8-hour-a-day user |
| 2 | **Alarm Fatigue Theater** | Generating alerts that satisfy audits but drive no action (400 alarms, 10 responses) |
| 3 | **Compliance-As-Retention** | Using audit trail dependencies and proprietary formats as switching cost |
| 4 | **Shadow System Inevitability** | When every user builds a workaround, the product is the compliance layer, not the tool |
| 5 | **Validation Tax** | Vendor updates trigger customer re-validation; shipping faster makes customer's life worse |
| 6 | **Seat-Count Mismatch** | Named-user pricing in broad/shallow usage patterns |
| 7 | **Data Roach Motel** | Easy in, structurally difficult out. No structured export. Customer's data requires a contractor to extract |
| 8 | **Monolith Misfit** | One surface for four fundamentally different jobs-to-be-done |
| 9 | **Power-User Ceiling** | No scripting, API, automation, or batch. Forces most capable users into most manual workflows |
| 10 | **Invisible Cost Accounting** | License fee is 40% of real cost. No vendor helps build TCO or ROI case |
| 11 | **Renewal Trap** | Auto-renewal, narrow cancellation windows, annual escalation, "AI feature" bundling |
| 12 | **Office-Floor Gap** | Software designed for desktop, deployed with shared terminals, gloved hands, 30-second windows |

### Language Mining (the market's words, not ours)

**From Dana (shop floor):**
- "Come stand at my press for one shift. Just one."
- "Data entry terminal" — how she describes her role in the software's eyes
- "Runs parts with no digital record" — what happens during downtime
- "Real data" vs. InfinityQS data — distinguishes without irony

**From Marcus (IT):**
- "That's not loyalty. That's a hostage situation."
- "Shadow SharePoint" — his name for the real system
- "InfoCard" — used with contempt; everything the same, nothing has meaning
- "600 hours a year" — the number the vendor doesn't acknowledge
- "2,000 hours to re-index exported PDFs" — exit cost that keeps him captive

**From Rachel (VP Ops):**
- "Paying for fighter jets to drive to grocery store"
- "I've told this story twenty times" — quantified word-of-mouth damage
- "You priced thirty-six out" — and "nobody complained" about the replacement
- "What does it cost to leave on day one?" — proposed purchasing criterion
- "Four sentences. Nobody says them." — the trust test

**From James (QE/Black Belt):**
- "Your competitor is a Black Belt with a laptop, four hours, and R."
- "Teaching tool pretending to be an industrial tool"
- "Analyses not performed" — the invisible gap
- "141 manual analysis runs, 6 dialog boxes each" — arithmetic of friction
- "Johnson transformation and a prayer"
- "Running analysis produces the auditable record" — what the tool should be
- "Validation theater" — what the tool is

**Recurring vocabulary:**
- "Workaround" — all four, always matter-of-fact, never embarrassed
- "Shadow" — the real system is always described as the shadow of the official one
- "Trapped" / "hostage" / "exit cost" — dominant metaphor is captivity
- "Nobody asks" — every persona; the vendor has never solicited their actual experience
