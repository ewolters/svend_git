# Focus Group: Canvas + Job + PCL Architecture Pressure Test

**Date:** 2026-05-06
**Technique:** Research-Grounded Focus Group (existing personas)
**Prompt:** Walk through your scenario again with this concrete architecture. Where does it help, where does it break, what's missing?

**Architecture tested:**
- Canvas = configurable screen, creates jobs on use
- Job = instance record (audit trail, institutional memory)
- PCL = shared data layer with typed events + policy
- Shadowboard = personal canvas arrangement, hotkeyed
- Org canvases = policy-defined standard screens
- Emergent workflows = promote repeated sequences to templates

**Personas:**
- Maria Kowalczyk — QE Manager, automotive stamping (Champion)
- Dave Przybylski — Quality Manager, aerospace job shop (Skeptic)
- Kenji Morales — CI Lead, food/bev packaging (Adjacent)
- Destiny Washington — Floor Inspector, plastics injection (Constraint)

---

## Maria Kowalczyk — Verdict: Strong Yes with Caveats

**What works:**
- PCL as shared data layer solving the 30-40% reformatting time waste — "If the capability canvas and control chart canvas both read the same PCL data, I'm not re-importing, not reformatting, not copy-pasting. That one feature alone is worth more than Minitab."
- PPAP auto-population from PCL — "That corrective action cost us $40,000. Because someone forgot to update a PDF. If PPAP reads from PCL, it literally cannot happen again."
- Shadowboard with hotkeys — "formalizing how I already think"
- Canvas as lens on same data — "I'm looking at the same thing from different angles. Right now those angles live in different files."

**What needs work:**
- "Job" language — "I'm not thinking 'I'm creating a job.' I'm thinking 'I'm looking at my chart.' Call it history. Don't call it a job to my face."
- Automatic job creation for glances — "I open my morning dashboard. That's a job? I'm just looking."
- Counterfactual provenance — simulated vs measured results must be distinguishable everywhere, automatically
- PCL overloading — "data events and policy events in the same system is conceptually muddy"
- Comparison mode — wants 3 scenarios side by side, not just one what-if

**Key quote:** "Don't make me learn your ontology. Make the ontology invisible and let me see my Cpk numbers faster."

---

## Dave Przybylski — Verdict: Conditional Yes, Starting from Response Letter

**What works:**
- Response letter pulling live data from analysis — "eliminating a failure mode I actually have. I have sent a response letter with the wrong Cpk. The irony of the quality manager having a quality problem with his own process is not lost on me."
- Hotkey shadowboard — "the first feature description in three years of looking at software that made me think 'yeah, I'd use that.'"
- Compare mode — "genuinely better than what I have"
- Audit trail searchable by part/date — "better than my folder system"

**What needs work:**
- Automatic job creation — "I look at charts forty times a day. Forty jobs a day is noise, not audit trail. Give me a 'Save as job' button. Not automatic."
- What-if as scratch work — "some auditor's going to pull up my what-if and ask why I ran a scenario that looks bad but never acted on it. Scratch work must stay scratch."
- PCL must be invisible — "If I have to 'manage' PCL, I won't."
- Offline/hybrid — "I will use Excel sometimes. That's not negotiable. The system has to accept that."
- Default canvases must be good enough — "if the answer is 'configure your canvases first' I'm closing the browser and opening Excel"
- Setup friction — "if I have to set up PCL first, configure canvases, define part library, establish characteristic taxonomy — I've got a customer complaint to answer and it's already 7:25"

**Key quote:** "Let me use one canvas for the one problem I have, and if it works, I'll use more."

---

## Kenji Morales — Verdict: Yes if Four Things Are Solved

**What works:**
- PCL propagation = "rebuilt the wall 4 times" fix — "Three rebuilds were because source data changed and I had to manually propagate. If PCL handles propagation, I rebuild zero times."
- A3 as lens over work already done — "The A3 isn't a document I write. It's a lens over the work I already did."
- Director walkthrough — "When the Director asks 'is this current?', the answer is always yes. That's the pitch."
- Canvas + Job identity — "better than a spreadsheet tab I named 'OEE_v3_final_FINAL2'"

**What needs work (four non-negotiables):**
1. **Manual data entry must be first-class** — "Half my data comes from a stopwatch and clipboard. If PCL treats manually-entered events as inferior to system-generated events, the architecture is built for a factory that doesn't exist."
2. **Observed vs projected must be explicit** — "Current state and future state need different event types or I'll pollute my own data within a week."
3. **Spatial arrangement matters** — "Composite canvas needs to mean 'I see five things on one screen, arranged how I want.' Not 'I configured a dashboard.'"
4. **Project-level grouping is non-negotiable** — "Without it, by month two I'm spending more time managing canvases than doing improvement work."

**Scaling concerns:**
- 3 months = 40-50 jobs across 8-10 canvas types. Needs: project view, archiving without breaking links, shared data sources across projects, template that captures workflow including PCL connections
- A3 must be selective reference (thinking tool), not auto-fill (report generator)

**Key quote:** "If those four things are solved, this replaces my conference room wall. If not, it's Smartsheet with better graphics."

---

## Destiny Washington — Verdict: I'll Believe It When I'm Standing at Machine 2

**What works:**
- Job as invisible timestamp — "I don't care about it, but I'm glad it's there. That's the system doing something useful behind my back."
- NCR pre-population from measurement — "The system already knows. I just told it."
- Package-tracking timeline = policy slots — "That is what I described. That's my FedEx tracker."

**What needs work:**
- "Flows through PCL" must mean real-time, same data — "In SAP they said 'integrated' and what they meant was a script that copied fields overnight. If PCL means real-time, same place, not a batch job — that changes things."
- Visible slots need actor tracking — "If QE Review sits there for three weeks with nobody touching it, what's the difference between a 'visible slot' and a task nobody did? I need 'QE Review — James opened this Tuesday.'"
- Speed is non-negotiable — "If I tap Machine 2 and it takes eight seconds to load because it's pulling from PCL and checking policies and building a job — I'm done. Paper takes zero seconds to load."

**Trust requirements:**
- One real end-to-end loop on real data — "Put my actual Machine 2 on it. Let me enter a measurement. Walk me to the QE's screen. Show me it's there. Right then."
- No "enter in both systems until migration" — "That's how it starts. Every time. 'Just for now.' Three years later I'm still entering it in both places."

**Key quote:** "Give me back the two other places I log that NCR. Give me the timeline so I stop walking to James's desk. Give me ten minutes of my morning back. Do that, and I'll train second shift myself."

---

## Synthesis

### Architecture Validated (4/4)

**PCL as shared data layer** — every persona independently confirmed this solves their core problem. Maria: eliminates reformatting. Dave: eliminates transcription errors. Kenji: eliminates rebuild-when-data-changes. Destiny: eliminates entering-in-three-places. Four different problems, one mechanism.

**Canvas as configurable screen** — accepted by all four without confusion. The concept maps to how they already think: "my chart screen," "my NCR form," "my A3 template."

**Shadowboard / hotkeys** — validated by the two power users (Maria, Dave). Irrelevant to Destiny (QE configures her screens). Kenji didn't engage with it directly but described composite canvas arrangement.

### Architecture Issues (convergent concerns)

**1. Job creation friction — 3/4 flagged this.**
- Maria: "I'm just looking. That's a job?"
- Dave: "Forty jobs a day is noise. Give me a button."
- Kenji: implicit in scaling concern (40-50 jobs in 3 months)
- Resolution signal: Jobs should be automatic but INVISIBLE. No naming, no modals, no "job created" messages. Silent logging. Explicit "save/promote" only for things you want to reference later.
- Alternative: distinguish between "views" (no record) and "runs" (creates job). Looking at a chart = view. Running a capability study = run.

**2. Observed vs. simulated/projected — 3/4 flagged this.**
- Maria: "projected Cpk also goes to PCL? That's a problem."
- Dave: "scratch work must stay scratch"
- Kenji: "Current state and future state need different event types"
- Resolution signal: PCL events need a provenance type. At minimum: observed, calculated, simulated, projected. This is non-negotiable for regulated environments.

**3. PCL must be invisible — 3/4 flagged this.**
- Maria: "Data and rules about data shouldn't live in the same bucket — or at least explain why"
- Dave: "If I have to 'manage' PCL, I won't."
- Destiny: "I don't need to know about 'jobs' or 'PCL'"
- Resolution signal: PCL is infrastructure. Users never see the word "PCL." They see their data, their results, their history. PCL is what makes it work, not what they interact with.

**4. Real-time, not batch — Destiny flagged, others implied.**
- Destiny: "In SAP they said 'integrated' and what they meant was overnight batch"
- Resolution: PCL events must propagate immediately. No sync delays. This is a trust requirement, not a performance preference.

### What Nobody Objected To

- Canvas as template / job as instance — universally understood
- Org-defined policy canvases — nobody pushed back
- Emergent workflow promotion — Kenji engaged most, others didn't object
- Typed events as connection mechanism — nobody found this confusing
- Audit trail as exhaust from normal work — universally wanted

### The Adoption Path (emergent from all four)

Dave said it most clearly: "Let me use one canvas for the one problem I have, and if it works, I'll use more."

Every persona described a single starting point:
- Maria: PPAP auto-population (the $40K corrective action)
- Dave: response letter that pulls live Cpk (the transcription error)
- Kenji: Pareto that updates when OEE data changes (the 4x rebuild)
- Destiny: enter once, goes everywhere (the 3x entry)

None of them asked for the whole system on day one. They each identified ONE canvas that solves ONE expensive problem. If that canvas works, they'll adopt more.

### Language

**Use:** canvas, run, check, analysis, history, trace, template
**Don't use:** job (internally fine, don't surface), PCL (infrastructure term), event, primitive, schema, policy (too abstract)
**Destiny's language wins for floor:** "Log NCR" not "create nonconformance event." "What happened last time" not "historical precedent query."
