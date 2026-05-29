# Focus Group: SVEND Rack Concept — Practitioner Reaction (Round 2)

**Date:** 2026-05-11
**Technique:** Research-Grounded Focus Group (same personas, new concept)
**Personas:**
- Dana Kowalski — Quality Manager, tier-2 automotive stamping (Champion)
- Raj Venkataraman — Sr. Process Engineer, medical device (Skeptic)
- Carmen Reyes — Independent MBB consultant/trainer (Adjacent)
- Tomasz Wojciechowski — Quality Technician, 60-person Polish CNC shop (Constraint)

**Round 1:** Reacted to current SVEND (200-option dashboard). 4/4 negative.
**Round 2:** Reacted to rack concept. 4/4 shifted to conditional engagement.

---

## Persona Responses

### Dana Kowalski

**First reaction:** "Meaningfully better. Not because it's flashy — because you stopped thinking about tools and started thinking about how work actually flows." The cable concept maps to her real workflow. "If the Cpk I ran on Tuesday lives somewhere and my control plan knows about it — that's the thing that's broken today."

**What clicks:** PPAP template pre-wired with capability→control chart→FMEA→control plan would save ~6 hrs/week of copy-paste. Back-of-rack view IS her audit trail — "when my IATF auditor asks how this Cpk ended up in this control plan, I flip the rack and show them the cable." Scratch vs persistent is exactly right. Green-into-green-only type system eliminates a class of inspector errors.

**Worries:** (1) Inspectors won't understand cable-building — templates must hide complexity. (2) Customer-specific output formats — Ford wants one Cpk report format, GM wants another. "Does it produce something I can hand to a customer, or something I still have to reformat?" (3) Stats must match Minitab exactly — "if there's a 0.01 difference in Cpk because you're using a different estimator, I have a problem." (4) Must replace Minitab, not add to stack.

**Templates:** Must be modifiable and saveable as her own. "Can I add a device, remove a device, re-cable something, and save that as my own template?" Annual revalidation: reload last year's setup, drop in new data, see what changed.

**Would pilot:** Yes. One part number, one PPAP, run in parallel with current process. Needs end-to-end flow with correct numbers and professional output.

**On the AI:** "Useful if it pulls up the last three capability studies for the same characteristic and shows me the trend. That's assistance. Interpreting my results is a parlor trick."

---

### Raj Venkataraman

**First reaction:** "The rack concept does something important: it decomposes the problem. Devices with typed ports — that's modular validation, fundamentally different from validating a platform." Would now pitch internally as technology to evaluate.

**Methodology enforcement:** "This is where you have my attention." Type system could enforce that capability study refuses to compute until data has been routed through a distribution fitting device first. "That's not AI. That's type enforcement. And it's exactly how you prevent the class of errors I spend ten hours a week catching."

**Critical pushback — semantic types:** Current 4-color system is too coarse. "A Cpk value, a p-value, a raw measurement, and a sample size are all 'numbers.' If I can cable a p-value into a slot expecting a Cpk threshold, your type system is decorative." Needs semantic types — "as rigorous as units in engineering." If done right: "you've built something that doesn't exist in JMP, Minitab, or any tool I've used."

**Validation:** Explicit cabling = dataflow graph = Design History File artifact. "Category change in audit trail quality." Templates could be validated configurations. But needs: (1) calculation transparency (algorithm, estimator, CI method visible per device), (2) NIST StRD verification per device, (3) 21 CFR Part 11 (role-based access, electronic signatures), (4) export/portability of rack configurations.

**Pitch to VP:** "Methodology enforcement with integrated audit trails that reduces review burden and improves FDA submission quality." $40K/yr loaded cost for 10 hrs/week review. "Build the semantic type system. Ship the NIST verification reports. Get Part 11 compliance. Then call me."

---

### Carmen Reyes

**First reaction:** "Better. Meaningfully better." Pre-wired DMAIC template maps to how she already teaches — "I literally draw arrows between boxes on a PowerPoint. You're telling me those connections are live and functional?"

**Template angle:** Strong positive. "If your template means 'do it once, it flows everywhere' — that's genuine workflow improvement." Condition: must build and share her own templates ("Carmen's Green Belt Template").

**Critical split — two modes required:** (1) Template mode = guided path with big clear steps, hidden cables — for students. (2) Build mode = full rack for practitioners. "If you try to split the difference and show cables to everyone, you'll lose my classroom in eleven minutes."

**License problem:** If free tier means student goes to URL, signs up, runs capability study in under 2 minutes with zero downloads — "you've solved my single biggest classroom logistics problem." Free tier MUST cover capability studies, control charts, basic hypothesis tests permanently. "If my student hits a paywall in week three, I will burn your name in every MBB Slack group."

**Curriculum rewrite:** Not on promise. Needs: (1) free 6-month account for one parallel cohort, (2) one other trainer who switched — phone call, not testimonial, (3) module-by-module migration over 2-3 cohorts, (4) Minitab data import (MTW/CSV).

**Key line:** "My fear is that you build a beautiful rack interface that impresses Black Belts at conferences and confuses the hell out of the people who actually need the most help."

---

### Tomasz Wojciechowski

**First reaction:** "Better. Clearly better." But concept vs execution: "you said modeled after Reason. That thing looked like the cockpit of a spacecraft." If he opens and sees cables and ports, first thought might be "this is not for me."

**Template angle:** "If I genuinely paste 50 measurements, type tolerance, and get a PDF I can email to my customer in Düsseldorf — then yes. That is faster than my Excel." His current workflow: 3 separate Excel files, copy-paste between them, screenshots into Word, format, export PDF = ~2 hours. Template in 15 minutes = immediate switch.

**Cables:** "Not for me as primary interaction." Would always start from template. Might learn to add one device occasionally. Templates are for him; cables are not.

**Trust — critical:** Must show METHOD, not just number. "If it showed me 'Cpk = 1.34 (within-subgroup, n=50, subgroup size=5)' — I could compare to what my customer expects. If it just shows 'Cpk = 1.34' with no explanation, I have the same problem as Excel except I paid for it."

**Threshold:** "50 measurements and a tolerance" to "PDF my German customer will accept" in under 20 minutes, first try, no documentation. Would switch from Excel immediately.

**Scratch mode:** "The thing that gets me to come back. The first time I use it and get the right answer in thirty seconds, I remember."

---

## Synthesis

### Consensus
All four validated the same core proposition: **pre-wired templates that map to real workflows are the entry point, not the rack itself.** Nobody led with cables or modularity. Each arrived at the rack through their existing workflow.

Additional convergence:
- Calculation transparency is non-negotiable (4/4 in different ways)
- Free tier must be permanent and functional (3/4 explicit, 1/4 implied)
- Scratch mode validated as behavioral hook for return visits (4/4)
- Templates must be saveable and shareable (2/4 unprompted)

### Divergence
Cable visibility split by expertise: Dana and Raj see cables as workflow/audit representation. Carmen and Tomasz see cables as internal complexity to hide. Driver is end-user expertise level — same concept, two interaction modes that must be cleanly separated.

Validation standard varies: Raj needs NIST + Part 11. Dana needs Minitab parity. Tomasz needs matching German customer's calculation. Carmen needs pedagogical clarity. Same demand (trustworthy numbers), four different proof artifacts.

### Strongest Objection
Raj: semantic type system. Current 4-color scheme allows methodologically invalid connections. Without semantic types, the rack is a nicer UI. With them, it's a category change. This is the objection that, if unaddressed, collapses the concept's strongest value proposition.

### Unmet Need
Customer-specific output formatting. Dana: Ford vs GM want different report formats. Tomasz: PDFs his German customer will accept. The rack produces analysis; the buyer needs it in their customer's document template.

### Would Pay
- **Tomasz:** Yes, immediately. Template works + numbers match = switch from Excel.
- **Dana:** Yes, as pilot. One part number, Minitab parity confirmed.
- **Carmen:** Not yet. Needs parallel cohort, peer validation, module migration path. Timeline in semesters.
- **Raj:** No, not now. Explicit checklist: semantic types, NIST, Part 11, export. Genuine but conditional.

### Shift from Round 1
- Dana: "Science fair project" → "meaningfully better" (conditional pilot)
- Raj: "Wouldn't pitch internally" → "would pitch as technology to evaluate"
- Carmen: "Science fair project" → "closer to my problem" (engaged skepticism)
- Tomasz: "Close the tab" → "genuinely yes if template works" (largest shift)

**The rack concept converted all four from rejection to conditional engagement. No unconditional adoption. Every condition is about proof, not concept.**

### Language Mining
"Modular validation," "flip the rack" for audit, "category change in audit trail quality," "parlor trick" for AI interpretation, "burn your name" for paywall friction, "same problem as Excel except I paid for it," "methodology enforcement," customer-specific formatting as procurement gate.
