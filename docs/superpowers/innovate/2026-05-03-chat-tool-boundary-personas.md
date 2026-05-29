# Chat/Tool Boundary — Persona Responses

**Date:** 2026-05-03
**Technique:** Targeted persona pitch (reused focus group personas)
**Question:** Where does the user interact with the tool directly vs. through Claude? What are the actual limits people expect?

## Personas Pitched

- Marcus Wade — CNC machinist, 2nd-shift lead, 45-person job shop (floor)
- Greg Linden — Quality Director, 340-person automotive stamping (office/management)
- Priya Chakraborty — Solo Quality/Compliance Manager, 95-person food plant (solo practitioner)
- James Okafor — Sr. QE / Black Belt, 800-person semiconductor (power user)

---

## Marcus Wade (Floor)

**Primary mode:** Option C, 90/10 weighted toward direct tool.

**At 10pm on a Tuesday:** A screen he can glance at — green/yellow/red, like glancing at the DRO. Has 90 seconds between pulling a part, mic'ing it, and loading the next blank. AI conversation maybe a few times a shift — between jobs, during setup, when he has a free minute.

**AI helps:**
- Borderline dimensions — "Is this tool wearing the same way it did on the last run?" (replaces digging through old paper travelers or waiting for morning)
- NCR dictation — "bore came in at .5017 on a .500 plus .001 nominal, I think the insert is chipped, pulled the tool" → AI writes the NCR properly
- Training new operators — historical data, trends, actual examples from their shop
- Root cause — "This Cpk has been dropping all week — what changed?" (connects tool changes, material certs, setup sheets)
- Disposition help — pull the print, show GD&T, show historical acceptance at customer
- End-of-shift handoff — describe what happened, AI generates proper handoff report
- "Something feels wrong" — "something's off on this bore, pull the last 10 readings"

**AI slows down:**
- Anything that replaces a glance (don't ask "what's my chart look like" — just show it)
- Follow-up questions when loading a part ("did you mean F-150 or F-250?" — system should know what's on the machine)

**Absolutely NOT through conversation:**
- Live SPC chart (always-on display)
- Alerts (flag pops up, done)
- Standard data recording (CMM auto-feeds, no confirmation needed)
- Part count, cycle time, basic status (dashboard, glanceable)

**Voice reality:** Barely. Integrex at full spindle = 85-90 dB. Ear protection. Coolant-soaked gloves. Doesn't want questions broadcast. Push-to-talk on rugged tablet with big buttons might work. Short commands, not conversations. "Flag last part." "Show bore trend." For most of the shift, tapping big buttons > talking.

---

## Greg Linden (Quality Director)

**Primary mode:** Option C, heavily weighted toward A. Engineers think in "open file, select column, run analysis."

**Quality engineers day-to-day:** Click. Button. File in, report out. Muscle memory from Minitab. Don't want to describe what they want — want to do it. AI earns its keep on second-order stuff: "Cpk dropped from 1.45 to 1.15 since last PPAP — what changed? Did we switch material lots? Tooling reworked?" Saves 2 hours of digging through SAP.

**58-year-old lead inspector (Dave):** Not talking to an AI. Needs screen with numbers, green/yellow/red, scan a traveler, see where the lot is. Chat window → he walks to a quality engineer. One exception: voice, single-button, "what's the reject rate on the 4140 housing this shift" — that's asking a question the way he'd ask Greg across the floor.

**Need to see and click:**
- Capability studies (must see histogram, data points, bimodal distributions — number without picture is worthless)
- PPAP package assembly (checklist, attach, approve — workflow)
- Measurement data upload (file in, confirmation out, see 50 measurements landed)
- SPC charting (visual, always visual)

**Conversation helps:**
- Investigation — "Why did this Cpk drop?" "Show me all CAPA actions for this part family, last 12 months"
- Meeting prep — "Top 5 scrap drivers for Line 3 this week with cost" (saves 45 min)
- Tribal knowledge — "What did we do last time the F-150 bracket went out of tolerance on the weld flange?"

**Absolutely NOT through conversation:**
- Anything going to the customer (PPAP, controlled documents, 8D, deviation requests)
- Measurement disposition (accept/reject with electronic signature — liability)
- Anything with audit trail requirements ("I asked the AI and it did it" = major nonconformance)

**7am scrap meeting:** Dashboard. Pre-built. Auto-refreshed. Nobody talking to AI at 7am with plant manager tapping fingers. AI helps AFTER the meeting — "show me burr rejects on Line 3 for 6 weeks overlaid with tooling change dates."

**Trust for PPAP:** No, not today. Would trust it if: (1) show the data it used (actual points, timestamps, sources), (2) show spec limits applied and where they came from (revision number), (3) lock the output (version-controlled document), (4) let him validate against Minitab the first 10 times. If it ever gives different answer than Minitab without clear explanation, trust goes to zero permanently.

---

## Priya Chakraborty (Solo QM)

**Primary mode:** Option C, heavily weighted toward forms for routine.

**Tuesday afternoon reality:** 4 minutes between interruptions. Click "New NCR," type lot number, pick "foreign material" from dropdown, save, move on. Does not want to compose a sentence. Does not want to verify AI understood correctly. Needs muscle memory — same three clicks every time.

**Investigation is different:** Thursday morning working the CAPA — "pull up last 6 months of foreign material complaints, show pattern by line or shift" saves 45 minutes across three Excel workbooks.

**SQF audit — auditor sitting across:**
- FIRST: search bar. Type lot number, get document tree (complaint → NCR → CAPA → root cause → corrective action → verification → training). Auditor sees it, she sees it. Looks professional and controlled.
- NOT: chat window while auditor watches. "Looks like I don't know where my own records are." Auditor evaluates whether SHE has a functioning system.
- Night BEFORE audit: conversation is gold. "Any open CAPAs past due?" "Which supplier CARs missing verification?" "Gaps in environmental monitoring last 90 days?" Audit prep in 1 hour instead of 3 evenings.

**Daily split:**
- Every day (log inspection, release hold, line check): clicking faster. Hands know where to go. Repetitive structured entry = what forms are for.
- Weekly/monthly (investigate trend, supplier scorecard, CAPA package): describing what she needs faster. That's where 12 hrs/week goes — not data entry, but synthesis.

**Trust boundaries:**
- Auto-link CAPA to NCR: yes (lot number matching, she'd see errors immediately)
- Draft root cause: she'll READ a draft, won't accept one (root cause is her expertise)
- Approve a document: absolutely not (her signature, her judgment, auditor checks this)
- Auto-close CAPA: no (must check effectiveness herself)
- Draft customer complaint response: yes, she'll rewrite half, but starting point with lot data saves 20 min

**The line:** AI can find, organize, draft, and route. Cannot decide, approve, or certify.

**Line leads:** Checkboxes. Gloves, floor, checks every 30 minutes. Big buttons: pass/fail, temperature, visual check, sign-off. If they have to type to an AI, they'll stop logging checks, and the recordkeeping gap is what the auditor finds.

**Weekend evaluation:** Evaluating the form-based interface. Can she create a CAPA in under 2 minutes? Link to complaint? See a dashboard? Set up document approval? If forms are clunky, no AI saves it. Conversation features are a bonus that moves SVEND from "comparable" to "this is the one."

---

## James Okafor (Power User / Black Belt)

**Primary mode:** C — programmatic API. No question. Secondary: API + conversation for investigation. GUI for Green Belts and auditors.

**Already has pipelines:** R scripts for 200+ parameters. Needs a validated computation engine he can call from existing pipeline structure. Swap the guts, keep the automation.

**AI helps genuinely:**
- Investigation: "Why did coplanarity Cpk drop?" — correlate across lot changes, equipment PM, incoming material shifts
- Edge case interpretation: "Bimodal distribution on die attach voiding — mixture of two processes or measurement artifact?"
- Report narrative: writing the management summary explaining what numbers mean

**AI is unnecessary translation layer:**
- Anything batch (don't describe it, call it)
- Standard monthly configurations (that's a script, not a dialogue)
- Anything where he knows exactly what he wants (don't compose English when `method="bayesian", distribution="weibull"` is unambiguous)

**200-parameter monthly run:** API call. Not even close.
```python
results = svend.batch_capability(
    data_source="mes_extract_april",
    parameters=ctq_list,
    spec_limits=spec_table,
    method="bayesian",
    distribution="auto",
    flag_below=1.33,
    output=["summary_table", "flagged_detail", "pdf_report"]
)
```

**Training Green Belt:** GUI + conversation. Green Belt needs to see histogram, spec limits, sliders. Conversation for "why is Cpk lower than Cp?" with their actual data.

**Investigating Cpk drop:** Conversation first, then API for follow-up verification. Trust but verify.

**Trust for AI-written code:** Needs to SEE it. Inspect what was computed. Needs full computation trace: method selection rationale, goodness-of-fit, confidence intervals. Black box "Cpk = 1.45" is worthless.

**Collapsing dual workflow (the purchase trigger):** Three requirements:
1. Statistics actually better than Minitab (Bayesian, non-normal, mixed effects)
2. Output audit-ready (control plan format, customer-reportable, IATF docs)
3. Programmable (API non-negotiable)

"The analysis you trust and the analysis the auditor trusts are finally the same thing, and you can automate it."

---

## Convergence

**4/4 independently drew the same line:**

| Direct Tool (Doing) | Conversation (Thinking) |
|---|---|
| Live charts, dashboards | "Why did this drop?" |
| Data entry forms, checklists | Cross-system investigation |
| Document approval workflows | Audit prep (night before) |
| PPAP assembly, customer-facing docs | Historical pattern retrieval |
| Batch/pipeline execution (API) | Report narrative drafting |
| Anything with audit trail | "What changed?" correlation |
| Anything with liability (disposition) | Root cause exploration |

**The architectural rule:** The app is the primary interface. Claude is available when the user has a question that charts and forms can't answer. Claude never stands between the user and the tool. Claude is the expert down the hall, not the receptionist at the front desk.
