# Session Architecture — Focus Group Pitch Draft

**Date:** 2026-05-03
**Status:** DRAFT — needs refinement before persona testing
**Input:** Integration constraints spec, chat/tool boundary resolution, screen design transfer, Eric's session-as-workstation concept

---

## The Pitch (Plain Language)

You open SVEND. You see a clean screen — what you were working on last time, already there. If you're new, it's empty.

You drop data in. A dataset, a CSV, a live connection to your measurement system. The data shows up.

You want to run capability. You either:
- Click the data column and pick "Capability" from a short menu
- Type `\cpk \usl 25.05 \lsl 24.95` in the command strip
- Tell Claude: "run capability on bore diameter, spec is 25 plus minus 0.05"

The result appears on screen — histogram, Cpk value, interpretation. It's a node in your session. The data flows from your source through the analysis to the result.

Now you want to monitor this ongoing. You add a control chart node downstream. Wire the same data column to an X-bar/R chart. Now you have two instruments — capability study and control chart — both live, both showing on your screen.

You want a report. You add a report node. It pulls the Cpk result and the control chart image, formats them into a template you configure once. Every time the data updates, the report updates.

You save the session. Tomorrow it's there. Next week it's there. The data is live. The results update. Your coworker can open it, see exactly what you built, exactly what data feeds it, exactly what every analysis shows. If the auditor asks "how did you arrive at this conclusion?", the session IS the answer — every step, every decision, every data source, visible and traceable.

That's one example. Simple. Three nodes. But the same pattern scales:

- Wire incoming inspection data through a changepoint detector, and if it fires, automatically trigger a capability study on the last 50 measurements since the shift, and if Cpk is below 1.33, draft a supplier corrective action with the evidence already attached.

- Open a value stream map alongside your process capability data, and see the actual cycle times and defect rates from your measurement system, not the numbers someone typed in six months ago.

- Build a Monday morning dashboard that shows last week's scrap Pareto, the top 3 capability risks, and any open CAPAs past due — and have it ready before you walk in.

Simple or complex. Single-use throwaway or permanent workstation. You build it visually, or with commands, or by telling Claude what you need. The session saves everything — what you built, why, what the data showed, what you decided. The audit trail writes itself.

---

## What This Solves (Mapped to VOC)

| Persona Pain | How Session Architecture Addresses It |
|---|---|
| Greg's CMM-to-PPAP copy-paste (20-30 min, 15x/week) | Data source node → Cpk node → report template node. One session, reusable, no copy-paste. |
| Priya's 45-minute audit trail reconstruction | The session IS the trail. Search by lot number, see every analysis that touched it. |
| Priya's 2-3 week audit prep | Saved sessions = pre-built evidence packages. Already formatted, already linked. |
| David's traceability gap between systems | Session graph connects SPC signal → CAPA → verification. One chain, not three systems. |
| Marcus's NCR black hole | His observation enters the session. The downstream nodes (CAPA, investigation) are visible in the same session. Status flows back. |
| James's 200-parameter batch run in Minitab (2 days) | One session with 200 capability nodes, each wired to a data source. Runs on schedule. |
| James's dual workflow (Minitab official, R real) | Session output IS the official record. Analysis and documentation are one artifact. |
| Dana's 4-7 minute data entry vs 30-second paper | Data auto-captured → session auto-populates. Zero entry for the common case. |
| Rachel's "what does it cost to leave?" | Session JSON is exportable. Your workflow definition, your data, structured and portable. |
| Marcus Chen's 600 hrs/yr maintaining integrations | No integrations. One system. Data flows through session nodes, not between separate applications. |

---

## Anti-Pattern Coverage

| Anti-Pattern | How This Prevents It |
|---|---|
| Buyer-User Split | Users build their own workstations. No one buys features they don't use. |
| Alarm Fatigue Theater | Alerts are nodes in the session — configured by the user, with downstream actions wired. Not random popups. |
| Compliance-As-Retention | Session JSON exportable. Workflow portable. Data exportable (TRUST-1). |
| Shadow System Inevitability | The session IS the real system. No reason to maintain a parallel spreadsheet because the session does what the spreadsheet did. |
| Validation Tax | Session definition is versioned. Re-validation scoped to changed nodes, not entire system. |
| Seat-Count Mismatch | Shared sessions. One user builds, others view or fork. |
| Data Roach Motel | JSON in, JSON out. Session definition + data + results all exportable. |
| Monolith Misfit | Not a monolith. A constellation of nodes, composed per user. |
| Power-User Ceiling | DSL, API, visual builder. James scripts his sessions. Marcus never touches a keyboard. Same architecture. |
| Invisible Cost Accounting | Session shows exactly what's running, what data it consumes, what it produces. |
| Renewal Trap | Session definitions are yours. Portable. Not locked to SVEND's runtime. |
| Office-Floor Gap | Marcus's session is one auto-populated chart. Priya's is a CAPA workflow. Same template, different composition. |

---

## Open Questions for Persona Testing

1. **Does "session" make sense as a concept?** Or does it need a different name? ("Workstation"? "Workspace"? "Flow"? "Project"?)
2. **Visual builder vs. DSL vs. Claude** — which entry point does each persona gravitate to?
3. **Sharing and forking** — Greg builds a session for the F-150 bracket. Can his quality engineer fork it for the F-250? How does that work?
4. **Complexity ceiling** — at what point does a session become too complex to understand visually? Is there a node count where the Alteryx problem (spaghetti pipelines) appears?
5. **The "Monday morning" pattern** — scheduled sessions that run before you arrive. Is that obviously valuable or does it feel like black-box automation?
6. **Report node** — is "build a template, wire data to it" intuitive? Or does it need Claude to do the formatting?

---

## Persona-Specific Pitch Angles

**Greg:** "You drop your PC-DMIS file in. You wire it to a capability study. You wire that to a PPAP report template. Save it. Next launch, same session, new data, new report. Thirty minutes becomes thirty seconds."

**Priya:** "Every CAPA you work lives in a session. The complaint, the investigation, the root cause, the corrective action, the verification — all in one chain. When the auditor asks, you pull up the session. Everything's there."

**David:** "The session definition is JSON. You can version it, diff it, validate it. When we ship an update, your sessions don't break because the node contracts are versioned. Here's the schema."

**Marcus:** "You walk up to your machine. The chart's already there. Data's coming in from the CMM. If something drifts, the flag shows up on the chart. You don't build anything. Someone set up the session once and it just runs."

**James:** "Here's the API. `svend.session.create(nodes=[...])`. Script your 200-parameter batch as a session. Run it monthly. The output is the audit record. One truth."
