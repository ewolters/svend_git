# Focus Group: Reach-Past Moments — Why Switch from What You Already Have?

**Date:** 2026-05-03
**Technique:** Research-Grounded Focus Group (quick mode — personas from prior anti-pattern research)
**Personas:** Dana Kowalski (floor user), Marcus Chen (IT gatekeeper), Rachel Torres (VP buyer), James Okafor (power user)

## Prompt

Why would you reach past what you already have (Excel, Minitab, legacy QMS, ChatGPT, consultants) for a new AI-guided quality platform called SVEND?

The competitive shelf:
- Excel — free, everywhere, already know it
- Excel + ChatGPT — paste data, ask question, $20/mo
- Minitab — known brand, auditor trusts it, $2,600/yr
- JMP — powerful, niche, $1,320-8,400/yr
- InfinityQS/SafetyChain — enterprise, already piped into the line
- A consultant — $2-5K/engagement, get the answer but not the capability

## Persona Profiles

### Dana Kowalski
CNC machinist / in-process inspector, 120-person auto stamping plant. 14 years on the floor. 7-minute data entry ordeal on legacy SPC terminal. Pencil-whips. Keeps paper shadow log because system lost data twice. When auditor comes, quality scrambles to backfill digital system from her notebooks. Current tools: Excel (self-taught X-bar/R from YouTube), legacy SPC terminal (compliance theater), composition notebook (truth). No budget, no authority.

### Marcus Chen
IT Manager, 400-person med device manufacturer. Manages 6 people, 3 validated systems. Every software update requires re-validation under 21 CFR Part 11. Administers 15 Minitab licenses ($39K/yr) — 6 used twice a month, 3 power users do 80% of work. Built shadow SharePoint that runs half of quality operations. Has killed more vendor pitches than he can count. Budget goes through him for anything with a server.

### Rachel Torres
VP Operations, 200-person contract mfg (aero/defense). Paying for two QMS simultaneously (migration "not done" at 18 months). Minitab auto-renewed for 3 years (missed renewal window). Lost best QE to retirement 4 months ago — new hire spending 40% of time asking questions nobody can answer. AS9100 audit in 6 weeks. Signs checks up to $50K without board approval. Evaluates on: headcount risk reduction, audit survival, exit strategy.

### James Okafor
Sr. Quality Engineer / Black Belt, 800-person semiconductor. Master's in IE. Runs 141 Minitab analyses/month (8-12 min each). Wrote R script that does same in 4 hours — told "we can't use R, it's not validated." Runs every analysis twice (R for truth, Minitab for official output). Asked for JMP — denied 7 quarters running. Three colleagues follow 2018 checklist and copy results without interpretation. Auditors ask "why this test?" and they freeze.

---

## Persona Responses

### Dana Kowalski

**First reaction:** "Another one." Tired of software bought for her, not built for her. AI — doesn't care. Needs something that doesn't lose data and doesn't take 7 minutes. Good opening line with "bring your ugliest dataset" but the pitch isn't the problem — Tuesday at 5:45 AM when the thing is spinning is the problem.

**Would make her care:** Can she log a measurement faster than her notebook? Under 15 seconds, no dropdown menus, no operator ID confirmation. When the auditor shows up, does this produce the paperwork or is Sharon from quality still calling at lunch to verify entries? Do the work once, both needs met. Tribal knowledge — gets why it matters (Rick retired, Line 4 progressive die knowledge walked out), but won't sit and type up what she knows. Side effect only.

**What would stop her:** Doesn't decide anything. No budget, not in the room. Plant manager hears "AI" and either gets conference-excited or says "we already have SPC." Trust — last system lost data twice. Notebook stays until proven. Can't afford downtime to learn.

**Needs to see:** Proof at her station, on her part, with her caliper. Not a conference room. Data still there tomorrow. One-week trial alongside notebook.

**Boss pitch:** "There's a tool that might let us stop doing data entry twice and have SPC records ready when the auditor shows up, cheaper than what we're not using right now." But probably wouldn't — "that's a quality department decision."

---

### Marcus Chen

**First reaction:** "Another one." Last demo said "AI-powered quality" 14 times, couldn't answer if audit trail is immutable or append-only. "Less than one Minitab seat" is a weird flex — cost isn't his problem. Validation is. Change control is. IQ/OQ/PQ for every system that touches quality data.

**Would make him care:** Let non-statistical people run capability studies with methodology baked in so they can't screw it up — gives back 3 days cycle time per investigation. Tribal knowledge of REASONING (not just data) — lost two QEs, couldn't answer FDA question without calling retiree. "Heard that promise before, always just a fancy search bar."

**What would stop him:** (1) 21 CFR Part 11 compliance matrix on first call or done. (2) AI black box — "the model suggested" is not a root cause, auditor needs reproducible analytical chain. (3) Another island to integrate/maintain.

**Needs to see:** Validation package from regulated customer. Try to break audit trail. Side-by-side on real CAPA with their data. Time savings quantified.

**Boss pitch:** Would NOT say "AI-guided." Would call it "statistical analysis and knowledge management platform." "Cuts quality investigation cycle time by 40%, reduces dependency on three people, validates cleaner than half what we already have."

---

### Rachel Torres

**First reaction:** "Another platform." Already paying two QMS + locked Minitab. "AI" slapped on everything now, means nothing. Three demos this year where AI was a chatbot that couldn't find its own help docs. Tribal knowledge hit nerve for two seconds before scar tissue.

**Would make her care:** Linda retired. Sarah (new QE) spending 40% of time on questions nobody can answer — why specific Cpk threshold on 7075 aluminum, which customer specs have undocumented requirements, why CMM drifts on 2nd shift (HVAC cycling, not machine). If Sarah could ask and get the answer Linda would have given with context of WHY — difference between passing AS9100 audit and major nonconformance.

**What would stop her:** (1) No migration within 90 days of audit. (2) Won't add cost without replacing something. CFO asking why $85K/yr QMS spend hasn't improved corrective action close-out rate. (3) Real cost is learning time + validation time + risk of running three systems. Need exit before entrance.

**Needs to see:** Reference customer in aerospace — phone call, not testimonial. Pilot on most problematic process (can Sarah answer Linda-questions?). Month-to-month with kill clause.

**Boss pitch:** Not yet. After pilot: "Costs less than one Minitab seat, solves single-point-of-failure since Linda left, generates auditor documentation. Run alongside one quarter."

---

### James Okafor

**First reaction:** Sounds written for VPs. Overhead. BUT "explains WHY an analysis was chosen" got attention. Spends 15 hours/month coaching colleagues on rationale for analyses they ran 6 months ago and don't remember. Audit prep is the real time sink.

**Would make him care:** (1) Ingest data formats without manual prep — MES, JMP-format, CSV dumps with headers that change every firmware update. (2) Actually understand semiconductor data — non-normal, unilateral specs, not standard playbook. If it confidently produces the wrong answer like his colleagues, worse than nothing. (3) Knowledge capture as byproduct only — tried SharePoint, wiki, lessons learned DB, nobody uses any of them.

**What would stop him:** (1) Software validation (IQ/OQ/PQ) — $40K/8 months for Minitab. No validation package = shadow IT forever. (2) Procurement inertia — denied JMP 7 quarters, not price, director won't sign PO for unknown vendor. (3) Political: adopting this proves director was wrong for 7 quarters. Some directors don't like that.

**Needs to see:** 141 analyses side-by-side with Minitab. Numbers match + better docs. Less-statistical colleague uses it independently and produces defensible analysis. Printed audit trail answering "why this test, what assumptions, what uncertainty" without reconstruction.

**Boss pitch:** "Costs less than one Minitab seat, generates audit documentation that took me two weeks to prepare last time — automatically." Would NOT mention AI, tribal knowledge, or colleagues' competence gaps.

---

## Synthesis

### Consensus Signals

**Universal "Another one" fatigue.** All four opened with dismissal. This is baseline emotional state, not skepticism to overcome. They have been burned repeatedly by software demoed in conference rooms that died on the floor.

**Nobody would say "AI."** Every persona stripped the word from their internal pitch. Marcus: "statistical analysis and knowledge management platform." James: wouldn't mention AI. Rachel: wouldn't mention AI. Dana: wouldn't pitch at all. The label is a liability.

**Audit documentation as universal hook.** Dana wants paperwork from same entry. Marcus wants reproducible analytical chains for FDA. Rachel needs auditor docs for AS9100. James spends 15 hrs/month on audit prep. Four industries, four regulatory regimes, identical pain: the work of proving you did the work correctly costs as much as doing the work.

**Tribal knowledge loss real, trust zero.** All four acknowledged the problem. All four followed with scar tissue. "Fancy search bar," "tried SharePoint, wiki, lessons learned — nobody uses any." They believe the problem. They do not believe anyone can solve it.

**"Cheaper than what we're already not using."** Dana, Rachel, James all framed value as displacement, not addition. Budget is not new money — it's money already wasted on shelfware.

**Proof on their data, their process, their station.** Nobody accepts a generic demo.

### Divergent Reactions

**"Fast enough" splits by role.** Dana: pencil speed, 15 seconds, zero clicks. James: zero data wrangling on messy formats. Different interaction models — point-of-capture vs. power analysis.

**Trust object differs by altitude.** Dana: data won't disappear. Marcus: audit trail immutable, analytical chain reproducible. Rachel: exit exists before entry. James: numbers match Minitab. Same word, four different things to prove.

**Regulatory burden varies by orders of magnitude.** Dana's plant: SPC records for auditor (low). Marcus: 21 CFR Part 11 formal IQ/OQ/PQ (high). Rachel: AS9100 with 6-week clock (urgent). James: semiconductor validation, $40K/8 months (prohibitive without pre-built package).

**Political exposure varies sharply.** Dana: none (no authority). James: negative (proves director wrong). Rachel: calculated (won't move within 90 days of audit). Marcus: institutional (shadow IT classification risk).

### Strongest Objection

**James's procurement inertia.** Not the loudest, but structural and product-unsolvable. "Been denied JMP for 7 quarters, not about price, about director signing PO for unknown vendor." He is the most technically qualified evaluator and the least able to act. His director has political incentive NOT to sign (validates 7 quarters of denial). This is a cold-start problem: unknown vendor can't get signed → needs reference customers → reference customers require someone to go first. No product fix exists.

### Unmet Need Discovered

**Analytical provenance — the chain of reasoning, not just the conclusion.** The prompt anticipated tribal knowledge capture. It did not anticipate that the primary value is defending past decisions under audit. James: "why this test, what assumptions, what uncertainty — without reconstruction." Marcus: couldn't answer FDA question about reasoning. Rachel: Sarah needs "the context of why."

The unmet need is a decision journal generated automatically as a byproduct of analysis. The auditor's question is never "what did you measure?" — it's "why did you measure it that way, and what did you consider and reject?" Nobody has a system that captures rejected alternatives and the reasoning that eliminated them. Every current system captures the conclusion and loses the deliberation.

### Would Pay

**Rachel** — most likely buyer. Budget authority, active pain (Linda + audit), stated pilot structure. Would pay if: aerospace reference exists, pilot succeeds, cost displaces existing line item. Buyer with a timeline.

**James** — would adopt as shadow tool first, formalize later. Would fight for PO if validation package exists and side-by-side is convincing. But "would pay" ≠ "can get PO signed."

**Dana** — never the buyer. No budget, no authority. User, not customer. Adoption path: someone else buys it, shows up at her station, faster than notebook or she ignores it.

**Marcus** — not a buyer, a gatekeeper. Kills purchases, doesn't initiate. Bar: validation package + Part 11 matrix on first call. Without those, conversation ends before features discussed.

### Language Mining

| Phrase | Source | What it reveals |
|--------|--------|-----------------|
| "pencil speed" | Dana | Real competitor is paper, not software |
| "do the work once, both needs met" | Dana | Dual-use: production task + audit artifact from single action |
| "fancy search bar" | Marcus | How market perceives every knowledge management claim |
| "reproducible analytical chain" | Marcus | Audit term of art — opposite of "the model suggested" |
| "cuts quality investigation cycle time" | Marcus | Time-based value, not capability |
| "reduces dependency on three people" | Marcus | Bus factor as purchase argument |
| "validates cleaner than half what we already have" | Marcus | Existing tools fail their own standard |
| "difference between passing audit and major nonconformance" | Rachel | Binary outcome — not "better," pass/fail |
| "need to know exit before I care about entrance" | Rachel | Exit cost evaluated before entry value |
| "single-point-of-failure since Linda left" | Rachel | Reliability language applied to people |
| "run alongside for one quarter" | Rachel | VP-acceptable trial structure |
| "headers that change every firmware update" | James | Real data ingestion problem no demo addresses |
| "shadow IT category" | James | Death classification for unapproved tools |
| "director signing PO for unknown vendor" | James | Actual bottleneck: vendor legitimacy, not product quality |
| "proves his director was wrong" | James | Adoption as political threat |
| "why this test, what assumptions, what uncertainty" | James | The three audit questions no tool answers |
| "without reconstruction" | James | Current state: rationale rebuilt from memory, not retrieved |
