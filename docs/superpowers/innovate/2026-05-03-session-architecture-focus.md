# Focus Group: Session Architecture Pitch

**Date:** 2026-05-03
**Technique:** Persona pitch test (reused integration focus group personas)
**Pitch:** Session-based workstation architecture — pipeline builder, crystallized views, analytical provenance as byproduct

## Persona Responses

### Greg Linden (Quality Director)

**First reaction:** "You have my attention." The traceability chain — raw data to Cpk to PPAP page — made him sit up. "Right now when Ford SQA calls, Karen spends half a day reconstructing how she got a number."

**Would engineers build sessions?** Karen would. Jason needs the first one built for him. Key concern: "wire it" and "add a node" sounds like visual programming. "My people are not programmers. They're quality engineers." Needs pre-built session templates: Cpk-to-PPAP, incoming inspection to disposition, scrap to Pareto. "If my people can open a template, point it at their data, and go — you've got something. If they have to build from scratch, you've got a tool for consultants."

**Wire data to analysis to report:** Mental model is right — they think in that sequence. But they think "steps," not "wires." Linear, not graph. "Show me the graph unless I ask for it" = only show graph view to power users.

**Naming:** "Session" means nothing. Would call it a **workbook**, a **job**, or a **setup**. "Pull up the F-150 bracket job." Dave calls everything a "setup."

**Saved scrap Pareto:** Valuable IF he can drill into data and correct miscoded entries. "An auto-refreshed chart that skips the human filter" needs the ability to intervene. Static picture from garbage data = fancy printout.

**Trial:** NOT on a launch. On the Escape rear crossmember — 3 years in production, mature, known Cpk values. Validate against Minitab history. "You never trial new quality software on a launch."

**Pitch to GM:** "Connects CMM data to PPAP reports. Eliminates copy-paste. Thirty minutes to five. Traceable record of how every number was generated." Would NOT mention AI, nodes, wires, or sessions. Would mention replacing Minitab seats ($14,400/yr) as the business case.

**Key line:** "When Dave — fifty-eight years old — drops his CMM file in and gets a correct Cpk on the first try without calling Karen for help. Show me that, and I'll find budget."

### Priya Chakraborty (Solo QM)

**First reaction:** "The CAPA chain thing — that's the pitch right there." The chain from complaint to training record all visible in one view. "That's not a feature. That's the reason I'd switch."

**Self-building chain:** Believes it IF she's making the links. "You connected A to B, and B to C, so here's A through C in one view — that's just good data architecture. That's not magic." Would NOT believe auto-detection of relationships. Wants to test: what happens when a chain branches (one NCR → two CAPAs)?

**Weekend evaluation:** Can evaluate in 3 hours, not set up real workflow. Would create one CAPA, one NCR, see if chain view works. "If that takes less than 30 minutes, you're in the running." First thing tested: data export. "If I can't get my data out cleanly, we're not having this conversation."

**Line leads:** "Should never see the word 'session.' Should never see the chain view." Need a form that looks like their paper form on a tablet. Entry becomes a node but they don't know or need to know.

**Naming:** "Session" sounds like therapy. Call it a **case**, a **record chain**, or a **file** (the digital manila folder). "Case file" works for auditors.

**Monday dashboard:** Valuable IF it's a saved search, not a configuration project. Must be changeable in under 2 minutes. "If changing the dashboard is fast, it's valuable. If it's a project, it's overhead."

**Pitch to VP:** "Links everything together so auditor gets one screen instead of four spreadsheets. Line leads see forms. I see the full picture. Cuts audit prep from two weeks to days." Would NOT mention AI. Would mention it once at end: "also has a search assistant that can flag gaps before the auditor finds them."

**Key line:** "The sell is the chain. The sell is forty-five minutes becoming two minutes. The AI is the bonus round."

### David Kwon (Manufacturing IT Manager)

**First reaction:** "First vendor pitch I've read where someone thought about the data model before the dashboard." Asks to see failure modes.

**Session JSON:** "Changes my integration calculus significantly." Diffable, Git-storable, code-reviewable. But how deep? Needs full state — every parameter, every threshold. "Byte-for-byte reproducibility of results given the same input data."

**Node contract versioning:** Needs to see: (1) actual versioning mechanism (semver per node type? pin to version?), (2) a real breaking change example with migration path, (3) deprecation policy in writing with contractual commitment. "If an update breaks a session, that's not a bug ticket — that's a deviation, a CAPA, and potentially a re-validation."

**FDA traceability:** "Genuinely good idea." Works IF links are machine-readable, not just visual. Pushback: "session graph only provides traceability for workflows that run inside this system." Needs source nodes that maintain provenance back to SAP, MasterControl record IDs. "CSV upload and hope for the best" doesn't cut it.

**Validation:** Session graph helps impact analysis — typed edges make it tractable. "If I change node 3 and the output schema doesn't change, I can argue nodes 4-7 are unaffected. Stronger than 'we looked at it and it seems fine.'" Needs: validated state flag, lock mechanism, side-by-side old/new during re-qualification.

**Integration:** "If this is system number eight, the answer is no." Needs to REPLACE at least one system, not sit on top of all of them. If it replaces SPC + analysis tooling = net reduction from 7 to 6 systems. If it's an orchestration layer needing Minitab underneath = added complexity.

**First sandbox session:** Incoming inspection — CMM CSV in, capability on 8-10 characteristics, auto-flag below Cpk 1.33, generate report. "I'd also intentionally feed it garbage."

**Pitch to VP:** Time, money, risk. "120 hours/quarter reconstructing traceability. $45K/year of my salary maintaining plumbing. Tool reduced inspection time from 45 to 10 minutes in sandbox. Pays for itself year one."

**Key line:** "The core value proposition is structural — versioned pipelines, typed contracts, traceable graphs — and the AI is just one of four ways to create the same artifact. Don't let your marketing team bury that."

### Marcus Wade (CNC Machinist)

**First reaction:** "That's actually close. You're showing me my chart. The one chart I actually need. That's the first time someone's pitched me software that sounds like it respects my time."

**Someone else sets it up:** "That's how it should work and how it never works." QE will get grouping wrong, wrong PC-DMIS routine, wrong check frequency. Needs ability to fix small things himself — "suggest a change" button or small adjustments. "I don't need to build the session. I just need to not be trapped when something's wrong."

**Auto-updating chart:** "That is THE thing." Currently 3-4 minutes per check (walk, CMM, write, plot, walk back). Auto-capture = 5-second glance. "Nobody's ever offered me faster than paper before." BUT: needs visible "last update" timestamp. Stale data is more dangerous than no data. "Make the whole screen border go orange" if data is old.

**Tapping the flag:** "A tap is fine. One tap." Has time between 4-minute cycles. But: "One tap, information appears, I read it in ten seconds, I tap anywhere to dismiss or it goes away on its own." The 5-line brief = exactly right. "Don't give me a statistics lecture." Suggestion "check insert wear" = helpful if right 70% of time. "Like having a good setup guy looking over your shoulder."

**NCR dictation:** "Absolutely." The reason he doesn't file NCRs isn't apathy — it's 15 fields on a system he uses twice a month at 10:30pm. Voice entry that creates a real NCR linked to chart data = "accountability without paperwork." But needs to see what it created before filing. 3-second confirmation.

**NCR status visibility:** "More valuable than I would have admitted." Currently doesn't care because loop was never closed. Same failure mode 3 months later, same NCR. "If I can see someone did something, I'll file more NCRs. Because I can see it matters." Don't make him go looking — badge on the session saying "NCR-247: resolved."

**Naming:** "Session" = therapist appointment. Would call it **setup** or **runsheet**. "Pull up the runsheet for the F-150 bracket." Or just "my screen." If forced to pick one word: **runsheet**.

**Would keep using after 30 days:** "If the CMM link works and the chart is always current, you couldn't take it away from me." The live chart is the hook. Everything else keeps him engaged. "If the CMM link is flaky, nothing else matters. Fix that first."

**Key lines:** "Nobody's ever offered me faster than paper before." / "Don't make me create an account with a password I'll forget. Badge tap or machine login."

---

## Synthesis

### Convergence (4/4)

**The chain/traceability is the purchase trigger.** Greg: "made me sit up." Priya: "that's the pitch right there." David: "genuinely good idea." Marcus doesn't care about the chain directly but his NCR status visibility is the same mechanism. The audit trail that writes itself — analytical provenance as byproduct — landed with every persona.

**Pre-built templates, not blank canvas.** Greg: "ship with pre-built sessions for things we do every week." Priya: needs a CAPA form that works in 30 minutes on a Saturday. David: "show me the incoming inspection session." Marcus: "someone else sets it up." Nobody wants to build from scratch. They want to point a template at their data and go.

**AI is the bonus, not the pitch.** Greg: would NOT mention AI to his GM. Priya: mention it once at the end. David: "don't let your marketing team bury" the structural value. Marcus: the chart is the hook, not the AI. Lead with the tool. AI is the expert down the hall.

**"Session" is the wrong word.** Greg: workbook, job, or setup. Priya: case or file. David: pipeline or workflow. Marcus: setup or runsheet. "Session" sounds like a login timeout or a therapy appointment. The word needs to come from their vocabulary, not ours.

### New Requirements Surfaced

1. **Pre-built templates** are non-negotiable. Cpk-to-PPAP, incoming inspection, CAPA chain, scrap Pareto. Users customize templates, not build from scratch.
2. **Stale data indicator** (Marcus). Visible timestamp, screen-level warning if data is old. Critical for trust.
3. **User-fixable config** (Marcus). Operator can adjust small things (wrong PC-DMIS routine, wrong frequency) without rebuilding the session.
4. **Data correction layer** (Greg). User can flag and correct miscoded entries, and corrections flow through the chain.
5. **Branching chains** (Priya). One NCR → two CAPAs. One complaint → three lots. Visual must handle non-linear chains.
6. **Validated state flag** (David). Lock a session version, side-by-side old/new during re-qualification.
7. **Node contract versioning with written policy** (David). Semver, deprecation timeline, contractual commitment.
8. **Source provenance through external systems** (David). Session source node must reference back to SAP/MasterControl record ID.
9. **No-password entry** (Marcus). Badge tap, machine login, shop network auth.
10. **Confirmation before filing** (Marcus). Show the AI-generated NCR for 3 seconds before submitting.

### The Name

Candidates from personas:
- **Setup** (Greg's Dave, Marcus) — "pull up the setup for that part"
- **Runsheet** (Marcus) — "pull up the runsheet for the F-150 bracket"
- **Job** (Greg) — "the F-150 bracket job"
- **Workbook** (Greg) — familiar from Excel
- **Case** / **case file** (Priya) — auditor-friendly
- **Workflow** (David) — technical, accurate
- **Pipeline** (David) — too technical for floor

Different personas would use different words. The system might need to support multiple names for the same concept, or choose one that works across all — "setup" is closest to universal but doesn't fit Priya's QMS world. "Workbook" might be the compromise.

### Pitch Refinement

The pitch works. Four refinements:

1. **Lead with templates, not blank canvas.** "Here are the five most common quality workflows. Pick one, point it at your data." Building from scratch is the power-user path.
2. **Lead with the chain, not the nodes.** The traceability story landed harder than the pipeline builder story. "Every step you take is traceable" > "wire nodes together."
3. **Show the linear view by default, graph view on demand.** Greg: "they think in steps, not wires." Show step 1 → step 2 → step 3 → output. The graph is underneath.
4. **Never lead with AI in the pitch.** It's the last thing mentioned, not the first. The structural value — traceability, templates, export, versioning — is the pitch.
