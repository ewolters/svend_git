# Focus Group: Session-Based Navigation Model

**Date:** 2026-05-05
**Technique:** Research-Grounded Focus Group
**Personas:** Maria (QE Manager, auto stamping, champion), Dave (QM, aerospace job shop, skeptic), Kenji (CI Lead, food/bev, adjacent), Destiny (Floor Inspector, plastics, constraint)

## Research Summary

Personas grounded via G2/Capterra reviews, Elsmar Cove Quality Forum threads, Software Advice adoption research, AIAG PPAP documentation studies, and IATF 16949 implementation challenge reports. Four segments represent genuinely different relationships to quality tooling: buyer-champion, skeptic-evaluator, adjacent-workflow user, and floor-level constraint.

## Persona Profiles

### Maria Kowalczyk — THE CHAMPION
QE Manager, Tier-2 automotive stamping, 340 employees, SE Michigan. IATF 16949. 5 Minitab seats ($13k/yr), 2 barely used. Engineers spend 30-40% of time reformatting data between tools. Ford SQE corrective action from PPAP revision mismatch — document sync failure, not process failure. Budget authority $8k/yr without VP. Burned by InfinityQS (5 months, $18k consultant).

### Dave Przybylski — THE SKEPTIC
Quality Manager, metal fabrication job shop, 85 employees, Midwest. AS9100. Runs quality alone + part-time doc control. Burned twice (ERP quality module = NCR log; Minitab = only he uses it). Built own VBA control charts in 2019. They work. 51 years old, CQM-OE, 14 years at plant. $0 software budget. Zero tolerance for implementation projects.

### Kenji Morales — THE ADJACENT USER
CI Lead, food/bev packaging, 500 employees, California. ISO 9001 + SQF Level 2. Black Belt (GE program). Runs DMAIC/A3/VSM. Tools: Excel, underused Minitab site license, Smartsheet. VSMs with sticky notes on wall. Rebuilt OEE→Pareto→5-Why→A3→Kaizen chain 4x because data changed. Would never search for "SPC software." Budget: $500/mo without approval.

### Destiny Washington — THE CONSTRAINT
Quality Inspector/Floor Lead, Tier-1 plastics injection, 220 employees, Tennessee. No degree, no belt. 9 years at plant. Does 100% of daily data entry. Logs same NCR in 3 places (paper, Excel, QMS). Knows the process better than QEs. Trained on 4 systems in 9 years. Not a buyer but her adoption determines whether the data pipeline works.

---

## Persona Responses

### Maria Kowalczyk

**First reaction:** Skeptical of "glass cockpit" framing but the rail concept immediately mapped to her problem. "That corrective action last year? That's exactly what you're describing, except broken." The phrase "data feeds Cpk, Cpk feeds SPC, SPC feeds PPAP report" — that's the thing.

**Would make her care:** PPAP package integrity. Same version of every number across capability study, control plan, and submission. When engineer updates Cpk, control plan flags "pending review" automatically. Also MSA coordination for 16 inspectors.

**Would stop her:** Price above $8k. Implementation engagement of any kind. Statistics that don't match Minitab exactly. Data ownership concerns (Ford/GM supplier security).

**Needs to see:** PPAP template on HER data (5 chars, gauge study, Cpk w/ CI, formatted report) in under 1 hour, no guided demo. One peer phone call (stamping/machining, IATF, similar size). PPAP export format Ford SQE accepts without reformatting.

**Pitch to boss:** "It keeps our PPAP data synchronized — same version of every number — so we don't get another document sync corrective action."

---

### Dave Przybylski

**First reaction:** Eyes glazed at "data bus" and "session chain." Thinks: "I've got a Cpk going sideways on titanium, I need to pull 30 subgroups and run the chart." Glass cockpit = "the pilot trained for 500 hours." Sounds like "something a smart person designed in a conference room."

**Would make him care:** Set up once, floor techs repeat without him. His macros do this but are brittle. If the rail IS the documented process for AS9100 audit — that's an audit artifact, real value.

**Would stop him:** $0 budget, VP sign-off required. Any IT involvement — must be browser, email login. "Configure from scratch" = scare phrase. His VBA macros work — bar is very high.

**Needs to see:** Blank template, 25 data points from real job, chart matching his Excel, Cpk matching, path to report. 20 minutes. Zero help. Reference customer <200 employees, AS9100, self-deployed. Phone call.

**Pitch to boss:** "Turns quality workflows into repeatable playbooks floor guys can run without me, audit trail built in." But not until after his 20-min test passes.

---

### Kenji Morales

**First reaction:** Split. "Glass cockpit and data bus" = engineer-software disguised as operational. But the left rail IS his VSM wall except the sticky notes actually pass data. Interested but heard this before.

**Would make him care:** OEE project rebuilt 4 times when source data changed. If downstream updates when source changes, and he shows Director a single chain-of-evidence artifact — real. Must speak HIS language: OEE, yield loss, changeover, DMAIC gates. Not Cpk/PPAP.

**Would stop him:** Price above $500/mo. Learning curve — if only he can use it, it's a "me-tool." Burned by two CI tools (prettier Word doc; needed dedicated admin). SQF data handling needs IT/QA signoff.

**Needs to see:** Demo running HIS project — line efficiency → waste Pareto → 5-Why → A3, auto-populated. Reference customer in food/bev or CPG. Free trial on one real DMAIC project.

**Pitch to boss:** "It's like if your DMAIC project folder was smart — when data changes, your A3 doesn't rebuild from scratch." Needs 90-second live demo or won't try.

---

### Destiny Washington

**First reaction:** Read it three times. "Data bus. Session chain. Glass cockpit. I work in a plastic parts plant in Tennessee, not a Boeing 787." But decoded: "pick your job, it sets up your tools in order" — that's her inspection rounds. Makes sense once translated.

**Would make her care:** Logs same NCR in 3 places. Doesn't trust any single system. Has no idea what happens to her data after entry — "logging into a void." If she could SEE her measurement flow into the next check, that would mean something.

**Would stop her:** Required fields she doesn't understand. Login timeouts. Not a buyer. Every system promised help, two made her job harder, one got discontinued.

**Needs to see:** Floor lead at similar plant doing actual morning routine. Real data. What happens when out-of-spec — does it tell her what to do next? Must work on tablet. Must handle NCRs her way.

**Pitch to boss:** "Sets up inspection tools in order, like a checklist that feeds the report — but I'd want to see if it handles NCRs our way."

---

## Synthesis

### Consensus (4/4 or 3/4 convergence)

1. **The rail concept is sound.** All four independently mapped it to something real — PPAP chain, audit artifact, VSM wall, inspection sequence. The underlying mental model works.

2. **Traceability over novelty.** Nobody cared about the session model as innovation. They cared about downstream outputs staying coherent when upstream data changes. The value proposition is data integrity across a chain.

3. **Self-service or nothing.** Every persona flagged implementation overhead as a kill condition. The implicit universal test: can I get real output from my real data without help?

4. **Peer reference > everything.** Maria, Dave, Kenji all named a phone call with a peer as the highest-trust signal. Not case studies. A person. Same industry, similar size, self-deployed.

### Divergence

| Dimension | Split | Driver |
|---|---|---|
| Terminology | Maria parsed instantly; Dave/Kenji/Destiny couldn't | Technical proximity to measurement systems |
| Configuration | Maria/Dave want it; Dave/Kenji fear being the bottleneck | Desire for output vs. fear of being the only builder |
| Price ceiling | $8k (Maria), $6k (Kenji), $0 (Dave), N/A (Destiny) | Different procurement paths, not willingness |

### Strongest Objection

**Dave: "Sounds like something a smart person designed in a conference room."**

This is a legitimacy objection, not a feature objection. No capability fixes it — only evidence does. Dave is the median quality manager. His 20-minute zero-help test is the real gate. The glass cockpit analogy became ammunition against the product ("the pilot trained for 500 hours"). Remove from all customer-facing language.

### Unmet Need Discovered

**Destiny: Closed-loop acknowledgment at point of data entry.**

The session model describes data flowing between stages — visible to whoever navigates the session. But Destiny only interacts with one stage. She never sees the rail. She gets no confirmation her measurement became part of the Cpk → PPAP → customer submission. That missing signal is why she logs in three places. Need: floor worker submits measurement, sees it land in the next check. Not an audit log — a visible "your data went here."

### Would Pay

| Persona | Buy? | Path |
|---|---|---|
| Maria | **Yes, conditional** | PPAP template on her data + Minitab-match + peer call → pitch to boss |
| Dave | **Not yet** | 20-min test → builds one session → floor techs use it → then maybe VP pitch |
| Kenji | **Possible** | Free trial → DMAIC project → 90-sec Director demo → AFE |
| Destiny | **Not a buyer** | But can kill adoption from below if tool creates friction |

### Language Mining

| Their phrase | Replaces | Why it's better |
|---|---|---|
| "Same version of every number" | "Data bus" / "connected by data" | Customer's actual pain in 6 words |
| "Repeatable playbooks floor guys can run" | "Power users configure, simple users..." | Frames output, not mechanism |
| "Chain of evidence" | "Audit trail" / "session log" | Legal/regulatory weight |
| "Logging into a void" | (no equivalent in prompt) | Names the floor feedback gap |
| "Me-tool" | (no equivalent) | Precise failure mode for configurability |
| "Document sync corrective action" | "Integration between tools" | Names the incident the product prevents |
| "Brittle" | "Manual" / "inefficient" | Dave's actual problem — fragility, not effort |
