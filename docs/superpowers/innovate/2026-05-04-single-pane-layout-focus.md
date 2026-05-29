# Focus Group: Single-Pane Layout Validation

**Date:** 2026-05-04
**Technique:** Research-Grounded Focus Group (reused personas from 2026-05-03/04)
**Personas:**
- Greg Linden — Quality Director, 340-person tier-2 automotive stamping (Michigan). Office power user, Minitab muscle memory.
- Marcus Wade — CNC machinist / 2nd-shift lead, 45-person job shop (Cincinnati). Floor, 90 seconds, gloved hands.
- Priya Chakraborty — Solo QM, 95-person food ingredients plant (New Jersey). Forms, interruptions, 14 Excel workbooks.
- James Okafor — Sr. QE / Black Belt, 800-person semiconductor (Minneapolis). Power user, scripts, 200-parameter runs.

## Prompt Tested

"SVEND uses a single view — one screen, not multiple windows or tabs for different tools. The top area is the instrument surface where your work appears (charts, VSM canvas, capability study, Hoshin matrix, calculators — whatever tool you're using). The bottom is a persistent command strip for controls and Claude interaction. When you switch tools, the top pane re-renders. You don't get side-by-side views of different tools simultaneously — it's one tool at a time in the main pane, with the option to have Claude's advisor panel slide in from the side when needed. Does single-pane switching work for your daily reality, or do you need multiple tools visible at the same time?"

## Persona Responses

### Greg Linden (Quality Director)

**First reaction:** Single pane is fine for most of what we do. Nobody tiles two Minitab windows. They open a file, run the study, look at it, move on. Concept isn't foreign.

**Where it falls apart:** CMM to PPAP. QE pulls up CMM report (PDF or CSV from Zeiss) and reads dimensions while populating the PPAP capability table. Needs measurement data visible while populating the study. Without that, she opens a separate window and copy-pastes between two applications — exactly the failure mode to eliminate. Same problem with 7am scrap meeting: drilling into a Pareto bar replaces the Pareto, loses meeting context.

**What works fine with switching:** CAPA tracking (linear — list, click into CAPA, work it, go back). Capability studies standalone. VSM (needs all the real estate anyway).

**What breaks QEs' workflow:** Anything that forces them to remember what was on the previous screen. Lisa has been doing this 11 years. Her workflow: data on the left, analysis on the right. Toggle back and forth = lose place = make mistakes = "this is worse than Minitab."

**Solution:** Pin panel. Not a whole second workspace. Let me pin one thing (data reference) in a side panel while the main pane runs the analysis. Like the Claude advisor panel, but for data.

**Pitch to Dave (58yo inspector):** "Dave, you know how on the Zeiss you've got one screen and hit the button for the next dimension? Same idea. One screen, pick your tool, see your thing. If you need to check something, pull up a reference panel on the side — like having a print taped to the wall next to your monitor." Dave gets that in 45 seconds. Single-pane actually helps with Dave — fewer windows = less confusion.

**Key line:** "Give me the pin panel and they'll come around."

### Marcus Wade (CNC machinist)

**First reaction:** One pane is the right call. Doesn't want side-by-side anything. 10-inch tablet bolted to Integrex enclosure, covered in coolant mist. Not dragging windows or pinching split views.

**What he actually looks at during a shift:** ONE tool, almost exclusively. SPC chart. 95% of shift. Pull part, mic it, point lands on chart, glance, see if drifting. Done. Not switching to capability study mid-run. Not opening VSM canvas at the machine. Those are office things, conference room things.

**Switching between views:** During normal production, should never have to switch. Default screen = live chart for loaded job, stays there until deliberate action. Problem: if the system MAKES him switch — Claude popup, notification takeover, navigation required back to chart. If anything knocks him off his chart and he has to tap twice to get back, that's a problem.

**One screen on a tablet:** Tablet is at arm's length. Hands wet. Nitrile gloves. 85-90 dB. Coolant mist film by hour two. Chart must be dominant. Command strip: how tall? 20% of 10-inch tablet = 20% less chart. Should be minimal — thin bar, maybe just a mic icon, expands on tap.

**90-second test:** Passes, with conditions. Chart is default view always. Nothing takes over screen without asking. Command strip thin enough not to shrink chart. Claude as flags not panels. CMM new point appears on chart already on screen — never routed to a "results" view to dismiss.

**Key line:** "The real question isn't the layout. The real question is whether the system understands that 'one tool at a time' actually means 'one tool, period, and it's always the chart.'"

### Priya Chakraborty (Solo QM)

**First reaction:** Single pane is better than fine — it's what she wants. One monitor. Tiling two Excel sheets makes both too small. "One thing at a time, big enough to actually use" is already how she works efficiently.

**When she uses side-by-side (and why):** Three scenarios: (1) Logging NCR, looking up supplier COA for material spec fields — data entry problem, not layout problem. If NCR form auto-pulls COA data, second sheet unnecessary. (2) Audit prep cross-referencing CAPA log vs training log — what she really wants is a REPORT that says "these 6 CAPAs are missing training verification." Doesn't need to SEE both lists, needs the ANSWER. (3) Complaint log + email draft — just multitasking.

**Real answer:** Side-by-side is workaround because Excel sheets don't talk to each other. If system handles cross-referencing, need disappears.

**Switching cost:** Depends entirely on state preservation. "Click a button and NCR form appears with data still there" = costs nothing. "Navigate menu, wait to load, re-enter filters, scroll to find place" = costs everything.

**4-minute interruption reality:** Single pane actually helps. One thing on screen. Knows exactly what she was doing. No "which of six open windows was I in?" moment. Needs state preserved when switching away and returning.

**80% ignorability test:** If command strip has 40 buttons for DOE/VSM/capability = fails. If she sees only NCR/CAPA/complaints/docs because system knows she's food safety QM at 95-person plant = passes.

**Hard requirement:** Inside a CAPA, need to glance at the original complaint. Don't make her leave CAPA, find complaint, navigate back. Link inline or pop in side panel. Slide-in advisor panel for linked records = the answer.

**Key line:** "I don't need two tools visible simultaneously. I need one tool that's smart enough to pull in context from the other tools when I need it."

### James Okafor (Sr. QE / Black Belt)

**First reaction:** Fine. Not the expected answer from someone running three monitors.

**Current multi-monitor setup:** Left = Minitab. Center = RStudio. Right = SharePoint/PDF specs. But almost never actually comparing two outputs pixel-by-pixel simultaneously. What he's actually doing: sequential with fast switching. Run study in R, glance left at Minitab's Ppk, look back. Three monitors exist because alt-tabbing in Windows is garbage and Minitab takes 40 seconds to bring a window to front with a big project open. Multi-monitor is workaround for slow software, not fundamental analytical requirement.

**What actually requires simultaneous viewing:** Two cases only: (1) Validation runs comparing SVEND vs Minitab to four decimal places — qualification activity, not daily. Does it once when building trust in a tool, then spot-checks. (2) Spec reference during analysis (customer spec limits, Cpk requirements) — reference material, not a second analytical output. Solved by command strip context.

**Does command strip compensate?** More than compensate. If he can script `batch.run(params=wafer_lot_42) | validate(against="minitab.csv") | report(format="customer")` — doesn't need to see intermediate steps. DSL makes multi-pane irrelevant for 90%. "The whole reason I stare at two screens is because I'm manually doing what a pipeline should do automatically."

**Deal-breakers:** (1) Pane switch latency > ~2 seconds on 200-parameter drill-down. In R he types `plot(results[[47]])` instantly. (2) Stateless switching — if drill into parameter 47, switch to 48, switch back to 47 and zoom/annotations gone = done. (3) No snapshot/freeze capability for later comparison.

**Key line:** "Single pane with a scriptable command strip is closer to how I WANT to work than my current three-monitor setup. My three monitors are a compensation mechanism for dumb software."

---

## Synthesis

### Consensus (4/4)

**Single pane is accepted or preferred by all four personas.** Nobody rejected it. Agreement clusters around three reasons:
- Floor reality kills multi-pane (Marcus tablet/coolant, Priya constant interruptions)
- Multi-monitor is compensation, not preference (James: "compensation for slow software"; Greg: Dave would prefer fewer windows)
- Linked context eliminates the need for side-by-side (Priya and Greg both: if the system cross-references, the reason for two windows disappears)

**State preservation is the universal requirement.** Greg, Priya, and James all independently named it as the condition for accepting single-pane.

### Divergence

Not accept/reject. What "single pane" must include:

| Persona | Single pane needs... |
|---|---|
| Greg | Pin panel for reference data alongside working tool |
| Marcus | Nothing alongside. Chart is home. Everything else is interruption |
| Priya | Slide-in panels for linked records |
| James | Snapshot/freeze capability and sub-second switching |

Greg and Marcus are the sharpest split. Greg's QEs want a reference pane; Marcus wants maximum simplicity. Not reconcilable with a single default — system needs to support both without making either feel compromised.

### Strongest Objection

**Greg's CMM-to-PPAP workflow.** Specific, recurring, high-stakes task that cannot be redesigned away. QE needs raw measurement data visible while populating capability study. Without pin panel, QEs open a separate browser window and copy-paste — defeating the integrated system entirely. If experienced QEs judge the tool as worse than Minitab for this task, adoption stalls at the most influential buyer layer.

### Unmet Need Discovered

**"Home base" as a concept.** The prompt described switching between tools. Marcus reframed: the issue is not how you switch, but what the system returns to. His SPC chart is the default state of reality during production — not one tool among many. Priya's state-preservation requirement is the office version of the same instinct. James's snapshot request is the power-user version. Underlying need across three personas: **anchoring**, not switching.

### Would Use This Layout

| Persona | Verdict | Condition |
|---|---|---|
| Marcus | Yes, immediately | Chart as default home, thin command strip, no unsolicited screen takeovers |
| Priya | Yes, and prefers it | State preservation across interruptions, slide-in panels for linked records |
| James | Yes, surprisingly | Sub-second switch latency, state preservation, snapshot/freeze |
| Greg | Yes for 70%, needs pin panel for rest | Without pin panel, QEs work around it and judge it inferior to Minitab |

### Language Mining

| Phrase | Source | The Prompt Didn't Use It |
|---|---|---|
| Pin panel | Greg | Distinct from side-by-side or split view — one fixed reference alongside one working area |
| Home base | Marcus | Default view the system returns to; resting state |
| Nothing takes over screen without asking | Marcus | Consent-based navigation |
| Smart enough to pull in context | Priya | Intelligence replaces layout |
| Floor interruption | Priya | 20-minute disappearance as normal, not edge case |
| Compensation | James | Multi-monitor as symptom, not solution |
| Snapshot/freeze | James | Temporal anchoring — saving view state for later comparison |
| Flags not panels | Marcus | Notifications as minimal indicators |
| Loses meeting context | Greg | Drill-down as destructive action during collaborative use |
| Sheets don't talk to each other | Priya | The indictment that forces side-by-side as workaround |

### Design Implications

Single-pane is validated. The risk is not the layout — it's the two features that make it viable:
1. **State preservation** (universal requirement) — without it, everyone struggles
2. **Pin/reference panel** (quality engineering requirement) — without it, the most influential buyer persona works around the system

Additional: home-base concept (floor default view), thin command strip (floor screen real estate), sub-second switching (power user retention), snapshot/freeze (power user comparison).
