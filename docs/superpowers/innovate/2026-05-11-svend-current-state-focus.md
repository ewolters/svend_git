# Focus Group: SVEND Current State — Practitioner Reaction

**Date:** 2026-05-11
**Technique:** Research-Grounded Focus Group
**Personas:**
- Dana Kowalski — Quality Manager, tier-2 automotive stamping (Champion)
- Raj Venkataraman — Sr. Process Engineer, medical device (Skeptic)
- Carmen Reyes — Independent MBB consultant/trainer (Adjacent)
- Tomasz Wojciechowski — Quality Technician, 60-person Polish CNC shop (Constraint)

## Research Summary

Sourced from Elsmar Cove forums (SPC software evaluation threads, free software threads, qualifying statistical analysis providers), G2 and TrustRadius Minitab reviews, FDA Process Validation guidance, JASP "Cost Effective Six Sigma" article, OPEX Resources curriculum, LinkedIn articles on Minitab pricing changes, Upwork/Fiverr LSS consultant listings, SPC for Excel pricing. Four segments identified: budget-authorized buyer in regulated manufacturing, statistically sophisticated skeptic in FDA-regulated environment, curriculum-dependent consultant/trainer with multiplier effect, zero-budget self-taught technician in emerging European market.

## Persona Profiles

### Dana Kowalski — Quality Manager, Tier-2 Auto Stamping
- 180 employees, $28M revenue, Grand Rapids MI
- Manages 2 CMMs, 8 inspectors, PPAP submissions, SPC on 40+ characteristics
- 3 Minitab licenses ($4,785/yr), IATF 16949 audit in November
- FMEAs in Excel (print, mark up, re-enter), control charts attached to PPAP via shared drive
- Burned by QI Macros, pitched Qualio/ETQ/MasterControl (document management pretending to do statistics)
- Controls $12K/yr software budget, under $5K self-approval

### Raj Venkataraman — Sr. Process Engineer, Medical Device
- Mid-size med device company, Minneapolis, ISO 13485
- ASQ CQE, 15 years experience, JMP ($3,400/yr)
- DOE on coating processes, process validation (IQ/OQ/PQ), FDA submissions
- JSL scripting for automated validation reports, Python for custom analyses
- Frustration: junior engineers producing bad analyses, spends 10 hrs/week reviewing
- Software validation per FDA CSA = 40-80 hours to switch tools

### Carmen Reyes — Independent MBB Consultant/Trainer
- Charlotte NC, 20 years manufacturing, formerly Fortune 500 chemical
- 4-6 Green Belt cohorts/year (8-12 students each), 3-5 client sites simultaneously
- Minitab ($1,595/yr), curriculum built around OPEX Resources textbook
- 30% of coaching time on license logistics
- FMEAs in Excel, A3s in PowerPoint, VSMs on butcher paper
- Burned by SaaS tool acquisition/shutdown, rewrote curriculum over Christmas

### Tomasz Wojciechowski — Quality Technician, Polish CNC Shop
- 60 employees, Wroclaw Poland, ISO 9001:2015, pursuing AS9100D
- 28 years old, mechanical engineering degree, self-taught statistics from YouTube and Elsmar
- Everything in Excel — templates, copied formulas, forum spreadsheets
- Once submitted Cpk 1.8 that customer recalculated as 1.2 (subgrouping error)
- Zero quality software budget, no purchasing authority
- German customers with strict data handling requirements

## Persona Responses

### Dana Kowalski

**First reaction:** "Here we go again." 200 analyses sounds impressive on a slide deck but she doesn't need 200 — she needs about 12 that work perfectly and produce output her customer will accept. "When I hear 200+ analyses, knowledge graph, synthetic data generation, Hoshin Kanri — that tells me this was built by someone who wanted to build everything, not someone who talked to me first. It sounds like a science fair project." The AI guide — she doesn't care. She needs a Cpk that matches Minitab, formatted the way her tier-1 expects it. "Leading with AI-guided makes me think you're selling the sizzle because the steak isn't ready yet." No guided entry with 200 options means she and her inspectors are lost.

**What would make her care:** The statistics take five minutes. The hours are lost in the connective tissue — exporting Cpk, pasting into Excel trackers, mapping to PPAP, cross-referencing FMEAs for 8D responses. "If you told me you run the capability study in Svend, it automatically links to the part number, the control plan line item, the FMEA, and the PPAP package, and when your customer calls about a complaint you pull up one screen — now you have my attention." FMEA that follows AIAG-VDA format exactly and isn't miserable to update.

**What would stop her:** Trust (IATF audit in November, burned by QI Macros), switching cost (years of historical data in Minitab project files), inspector usability, and ironically "free" makes her nervous about longevity.

**What she'd need to see:** Her own data run side by side with Minitab. One automotive reference customer who replaced Minitab and survived an IATF audit. Her real 47-line PFMEA working in the tool.

**Pitch to boss:** "Platform that could replace our Minitab licenses and our Excel tracking mess for less money, but I need to run a pilot on one part number." But only if SPC-PPAP-FMEA connectivity exists. If it's just statistics: "I wouldn't bring it up. I'd close the tab."

---

### Raj Venkataraman

**First reaction:** 200+ analyses is a warning sign — "not a feature, that's a warning sign." Questions depth. "When someone tells me they do everything, I assume they do nothing well." JMP does statistics extraordinarily well and doesn't try to be his FMEA tool. AI guide: "I've seen this movie. Minitab Assistant guides people into running a 2-sample t-test when they should be running a paired comparison." In a regulated environment, wrong AI recommendations are dangerous — "false confidence."

**What would make him care:** Junior engineers running capability on non-normal data without transforming, wrong hypothesis tests, bad DOE with heteroscedastic residuals that almost went into a validation protocol. "If your tool could actually prevent that — not by dumbing things down, but by making the methodology transparent and enforced" — automatic normality testing, distribution fitting with AIC, non-normal capability via Clements method, all documented for validation protocols. That solves a real problem.

**What would stop him:** Software validation (40-80 hours per FDA CSA), IT security (SOC 2, BAA, data processing agreement), AI nondeterminism concern, no track record, no published computational methods. "Low price signals startup that might not exist in two years."

**What he'd need to see:** NIST Statistical Reference Dataset verification. Side-by-side JMP comparison on non-normal capability. DOE with lack of fit diagnostics. 21 CFR Part 11 compliance. Real medical device reference customer with FDA submission using this tool. One hour with the actual computation engine running his own data.

**Pitch to boss:** Wouldn't. Not yet. If everything checked out: "New platform costs a third of JMP and has methodology guardrails that would reduce my 10 hrs/week reviewing junior engineers' work." The hook is his time back, not price.

---

### Carmen Reyes

**First reaction:** "You lost me right there" at 200 analyses. "I teach Green Belts. These are plant supervisors and quality techs who just learned what a p-value is last Tuesday. They need capability study, Xbar-R chart, 1-sample t, 2-sample t, paired t, one-way ANOVA, chi-square, simple regression, and maybe binary logistic. That's like twelve things." 200 options = closed tab. "I hear a tool built by engineers for engineers... That's not a product, that's a science fair project."

**What would make her care:** One link she sends to 10 students from 4 companies. They sign up in 2 minutes, run a capability study before Thursday. That solves a problem she has every cohort. Also: FMEA + A3 + VSM in one place for kaizen events, accessible from phone during gemba walk.

**What would stop her:** Got burned by SaaS tool that got acquired and shut down (rewrote 4 weeks of curriculum over Christmas). Switching cost (200 slides reference Minitab screenshots). Credibility (students Google tools). Her own learning curve.

**What she'd need to see:** Permanent free account (not trial). One other trainer who switched. Connected FMEA-to-control-plan.

**Pitch to colleague:** "Web-based stats tool, cheaper than Minitab, might have too much going on, could solve the license headache." Less generous: "Startup crammed every analysis into a website. We'll see if they're still around next year."

**Key line:** "You're leading with your technology instead of my problem. My problem isn't 'I need more analyses.' My problem is 'I need ten people in a room to all be able to run the same capability study in the next five minutes without an IT ticket.' Solve that and I'll listen to everything else."

---

### Tomasz Wojciechowski

**First reaction:** Uses maybe 6 of 200 analyses. "Knowledge graph. Hoshin Kanri. Synthetic data generation. I don't even know what half of these words mean." Would open, not find "Capability Study" in 30 seconds, close tab. "I have done this before. Literally this exact thing. Twice."

**What would make him care:** 50 measurements, one tolerance, Cpk + control chart + professional PDF for German tier-1 auditor. "If your tool can do that — take my 50 measurements, ask me 'what is the tolerance?', give me Cpk, normality check, control chart, and a PDF I can email — and I can do it in five minutes without reading documentation? Then I care very much." Also: trustworthy Gage R&R (his Excel spreadsheet may have errors, customer flagged his %GRR as "unusual").

**What would stop him:** $49/mo is borderline — boss sees quality software as "Tomasz wants a toy." German customer data handling vs US cloud hosting (binary blocker). 200-option overwhelm. AI guide buried in menu.

**What he'd need to see:** Paste measurements, type tolerance, click go. Three-minute video of someone like him. Real free tier (been tricked by paywalls). EU hosting.

**Pitch to boss:** "It replaces Minitab for what we actually need, costs fifty euros per month instead of two thousand per year, and the capability reports look professional enough for customer audits." But only if he can trust the numbers.

**Key insight:** "Your real competition for someone like me is not Minitab. It is the Excel template I downloaded from Elsmar Cove in 2024 that is probably wrong but is free and I already know how to use it. Beat that."

---

## Synthesis

### Consensus
All four independently rejected 200+ analyses as a negative signal, not a feature. Three of four converged on needing 6-12 analyses. All four identified the connective tissue between tools — not the tools themselves — as the real gap. All four raised survivability/trust concerns about a startup.

### Divergence
Price perception split by context: Carmen sees $49 vs $1,595 as compelling; Tomasz sees $49 as borderline; Raj sees low price as credibility problem; Dana didn't mention price. AI positioning landed flat across all four — glazed eyes, regulatory concern, ignored, or buried. Regulatory burden creates hard segmentation: Raj's 40-80 hour validation is a different universe from Tomasz pasting 50 measurements.

### Strongest Objection
Carmen: "You're leading with your technology instead of my problem." This matters most because Carmen is the multiplier — she puts tools in front of 50-70 students/year. Her objection is about orientation, not features. She would be easiest to win (lowest switching cost, strongest price motivation, highest distribution leverage), making failure to reach her especially costly.

### Unmet Need Discovered
Raj: his problem isn't doing analysis — it's reviewing junior engineers who produce bad analyses. 10 hrs/week. He needs methodology enforcement, not a better statistics tool. Reframes value from "do statistics" to "prevent statistical malpractice."

Tomasz: EU data residency. German tier-1 data handling clauses vs US cloud is a binary blocker.

### Would Pay
- **Carmen:** Closest. Contingent on permanent free tier, 2-minute onboarding, one peer reference.
- **Tomasz:** Would adopt free tier if Cpk found in 30 seconds. $49/mo requires boss approval unlikely to get.
- **Dana:** Would pilot one part number if SPC-PPAP-FMEA connectivity exists and one auto reference available.
- **Raj:** Would not adopt now. Validation burden, missing compliance certs, no track record create wall no feature set addresses.

### Language Mining
Phrases the personas used that the prompt did not: "connective tissue between tools," "Excel tracking mess," "science fair project," "do everything, do nothing well," "the license headache," "Tomasz wants a toy," "startup crammed every analysis into a website," "reducing my 10 hours a week reviewing," "survived an IATF audit," "paste measurements, type tolerance, click go," "the Excel template I downloaded from Elsmar Cove in 2024 that is probably wrong but is free and I already know how to use it."
