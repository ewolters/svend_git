# Focus Group: SVEND Container + Marketplace Model

**Date:** 2026-05-04
**Technique:** Research-Grounded Focus Group (reused personas from 2026-05-03)
**Personas:**
- Maria Gutierrez — QM at tier-2 auto stamping, 180 people (Champion)
- Dave Kowalski — CI Director at med device contract mfg, 1,200 people (Skeptic)
- Tameka Jackson — FSQA Manager at snack co-packer, 90 people (Adjacent)
- Ray Nguyen — CNC job shop owner, 55 people (Constraint)

## Prompt Tested

"SVEND container + marketplace model — pick and install quality apps from a marketplace, Claude AI as subtle facilitator, provenance trail, knowledge retention, data export, visible contracts between apps. Questions: what would you pay, what does onboarding look like, what support do you expect, what apps first?"

## Persona Responses

### Maria Gutierrez (Champion)

**First reaction:** "App store" framing resonated — tired of paying for 80% she doesn't use (Minitab $2,400/yr, uses 20%). Nervous about "AI watches you" — inspectors will read it as surveillance. WiFi constraint real. Knowledge retention "actually stopped me" — lost 2 QEs, lost control limit rationale for B-pillar line.

**Pricing:** $300-400/mo site-wide. NOT per-user — "the minute I have to count seats, inspectors don't get it and I'm back to Excel." Free tier to try. Annual pricing with monthly eval option. Comparison math: Minitab $200/user/mo for Connect, InfinityQS $15K+ implementation.

**Onboarding:** 4 hours personal over 2 weeks. First characteristic charted in 45 minutes or lost. Inspectors: 2-hour video training max, tablet-ready. Needs setup call with someone who understands automotive quality — "not a generic customer success person reading a script." Will NOT migrate 14 Excel workbooks — starts fresh and lets old data age out.

**Support:** $300-400/mo = email within 1 business day + automotive-relevant knowledge base. $500+ = phone or live chat. Hates ticket routing. Wants user community of other stamping/small mfg quality managers.

**Apps first:** SPC charting (stops the audit bleeding) → capability analysis (feeds SPC setup + PPAP) → CAPA tracking (paper log is embarrassing). Skip: DOE (no staff), check weigher (not packaging). Missing: gage R&R (core IATF requirement, does it 4x/year), PPAP documentation support, CMM data linkage.

**Stops her:** Price creep (free→$800/mo). No automotive reference customer she can call. WiFi dependency. Political risk — "if I bring something in and it fails, that's on me."

**Boss pitch:** "It's like an app store for quality tools with a built-in audit trail — we start with SPC to fix the audit finding, pay only for what we use, and the system keeps the process knowledge when the next QE leaves."

---

### Dave Kowalski (Skeptic)

**First reaction:** "I've seen this movie. ETQ tried module system too." AI suggestions that are statistically wrong = trust destroyed permanently — "not reduced, gone." But provenance trail: "that one I actually paused on." Price "suspiciously low" for what's promised.

**Pricing:** $12K-18K/yr site-level (4-6 QE users + him). Must demonstrably replace Minitab seat cost or he's paying twice with more risk. Flat site license, not per-user. 90-day pilot with real data before signing — "clock doesn't start until my data is in the system." VP will ask "what are we getting rid of?"

**Onboarding:** 2 hrs/week max. Needs someone who understands 21 CFR Part 820, not a QMS checklist reader. Pre-built CAPA/NCR/capability templates ready to use, not blank canvases. DOCUMENTED IQ/OQ/PQ validation package — vendor-provided, not self-written. Modular 30-minute role-based training. Must run Ppk study on known dataset and match Minitab exactly — "not 'let's discuss the methodology' done. Done."

**Support:** At $15K/yr: named contact (not ticket queue), same-day response during business hours, under-1-hour for audit emergencies, after-hours emergency number. "Submit ticket, 2 business days" = will not renew. Has been burned by this before.

**Apps first:** SPC, capability, CAPA, NCR/defect log, scrap Pareto — "table stakes." Interested in DOE (validates every result against Minitab first). Will turn off AI suggestions by default — "if it's on by default and I can't disable it, that's a problem." Missing: Gage R&R (should be Tier 1 — "if it's a separate add-on, someone made a bad product decision"), spec/tolerance management across part families.

**Stops him:** Validation burden (every app update = revalidation event?). One wrong statistical number = done permanently ("found out in a customer audit, that conversation sucked"). Switching cost perception. AI line between suggesting templates vs making statistical decisions — "I need to show an FDA inspector exactly what the system did and why."

**Boss pitch:** "It's a modular quality analysis platform that captures our engineers' decision-making so we stop losing institutional knowledge every time someone leaves — and I want to run a 90-day pilot before we commit to anything."

---

### Tameka Jackson (Adjacent)

**First reaction:** "This sounds like it was built by engineers for people who have time to think." But marketplace = respect (pick what you need, not paying for SafetyChain features she's never touched). AI learning = nervous ("you have to feed the machine for six months before it does anything useful"). Provenance trail = "NOW that I hear" — got dinged on CA traceability, has spiral notebook.

**Pricing:** $250/mo ceiling (self-approve). $300 = pushing it. Over that = GM memo that probably dies. NOT per-user — "per-user pricing at a 90-person plant sounds like $2,000/month to a GM." Comparing to SafetyChain $49/mo. Adoption path: free tier with check weigher → get GM a number → then ask for budget. "If I'm paying $200-250, this needs to do something SafetyChain can't do. The bar is not high, but it has to clear it."

**Onboarding:** Useful in under 30 minutes day one. "Not 'useful' like I completed the setup wizard. Useful like: I uploaded my check weigher CSV and I see a chart I can screenshot and show someone." Cannot do 4-hour implementation call. Will not watch 45-minute video series. Needs: "Upload your data here" button as first thing she sees, result in plain language (not "Cpk = 0.87" — tell her "your fill weight is drifting high on Line 3 between 10am and 2pm"), one 20-minute call where a human looks at her data. Trains nobody for 30 days — uses it herself first.

**Support:** $200-250/mo = email within 1 business day, human reads it (not auto-reply + ticket number). Phone for food safety emergencies — hard line. Community forum of other FSQA people. "Honestly more useful than a help desk half the time."

**Apps first:** Check weigher/fill weight analysis (day one — "only thing that can pay for itself fast enough for my GM to care"), CAPA tracking (fixes audit finding — needs owner field, due date, status, photo attach). Skip: DOE ("I don't run experiments, I run a production line"), FMEA (consultant does it every 2 years), scrap Pareto (not first problem). SPC charting: maybe, if connected to check weigher automatically. Missing: supplier nonconformance log, auditor-ready CA report generator.

**Stops her:** Price creep ($49→$340 by contract time). Data lock-in. Setup > 1 week. Breaks during audit. First result doesn't make sense = abandons. "If the first screen after I log in is asking me to configure integrations or set up an org chart, I'm done."

**Boss pitch:** "It's like an app store for food quality tools — we pay for what we use, and the first thing I want to try is the fill weight analysis, because I think it'll show us where we're giving away product."

---

### Ray Nguyen (Constraint)

**First reaction:** "When I hear 'marketplace,' I hear 'we'll charge you for things you used to get included.'" Burned by CAM vendor 40% price hike. AI "learns workflows" — "does it know my customer Boeing has a 1.67 Cpk requirement and my Lockheed job runs to tighter print tolerance than what's on the drawing because my program manager negotiated it verbally three years ago?" Audit trail: "NOW you have my attention" — AS9100 in 4 months.

**Pricing:** $200/mo = buys without thinking. $500/mo = prove ROI in 60 days with specific dollars. Math: 20% scrap reduction = $1,200-1,600/mo saved, so $500 pencils out. Flat monthly per site, month-to-month before annual commitment. "After what happened with my CAM vendor, I won't sign annual until I've lived with something for 90 days." Free tier apps must be actually useful — "if it's a billboard for the paid version, I'm gone."

**Onboarding:** 4 hours total. Upload CSV or connect MeasurLink, something useful within first session. Lead machinist won't read a manual — "if I have to explain it to him more than once, it's not getting used." Does NOT want paid implementation consultant. Wants documentation written for a job shop, not a Tier 1 supplier with a quality department. "If the first interaction is a 45-minute demo call before I can touch the product, I'm already skeptical."

**Support:** $200/mo = 24hr ticket response + docs + community forum. $500/mo = same-day + someone who knows AS9100. Will not tolerate chatbot bouncing. Does NOT want monthly check-in calls or dedicated CSM. "Just answer my question when I have one."

**Apps first:** SPC charting (short-run capable — CUSUM, Laney U-chart, something for 25-piece lots), capability analysis (reports Cpk to Boeing and Lockheed — "needs to be right"), scrap Pareto (if connected to SPC data, not just a bar chart), CAPA tracking (audit trail for AS9100). Evaluate FMEA at 60 days. Skip: DOE (no volume for experiments), check weigher (not his world), basic Shewhart (already has MeasurLink — "if your control chart app is just a prettier version of what I already have, I'll skip it"). Missing: FAI workflow tied to part number + CMM output, print tolerance management for verbal/email customer negotiations.

**Stops him:** Useless data export (JSON blob vs real CSV his CMM software can import). Lock-in creep (free apps migrating to paid tiers — "AS9100 shops talk to each other"). Generic AI suggestions ("have you considered reviewing your control plan" = cancel same day). Onboarding that starts with demo call before product access.

**Boss pitch (to lead machinist):** "It's like MeasurLink but it actually tells you when something's going wrong on a short run, and it keeps a paper trail so the auditors don't give us grief."

---

## Synthesis

### Consensus (4/4)

1. **"Pay for what you use" is the frame they all reached for.** The prompt said "pick and install" — every persona heard "pay for what you use." This is the marketplace's value proposition in the market's own words.

2. **Flat site license, not per-user.** Universal dealbreaker. Per-user triggers comparison math against Minitab and creates political friction with management ("sounds like $2,000/month").

3. **SPC and capability are table stakes.** Every persona named SPC first, capability second. These are the minimum viable product.

4. **Gage R&R is missing and everyone noticed.** Maria, Dave, and Ray all volunteered it unprompted. Dave said it should be Tier 1 — "if it's a separate add-on, someone made a bad product decision."

5. **Provenance/audit trail was the one AI feature that landed.** AI-as-facilitator made 3 personas nervous. Audit trail made 3 personas pause and lean in. Same technology, radically different reception based on framing.

6. **Knowledge retention is a purchase trigger.** Maria (lost 2 QEs), Dave (35 years institutional knowledge gone), Ray (verbal tolerance agreements). Tameka less affected but she's the solo practitioner — she IS the knowledge.

7. **Data export is a trust signal, not a feature.** Ray: JSON blob vs real CSV. Tameka: lock-in fear. Dave: audit documentation. Export communicates "we won't hold you hostage."

8. **First session must produce something real.** Tameka: 30 minutes. Ray: first session. Maria: 45 minutes. Dave: pre-built templates from session one. The exact times differ; the expectation is identical.

### Divergence

- **AI facilitation is four different objections:** Maria = surveillance optics (inspectors), Tameka = utility timing (6 months?), Ray = contextual blindness (verbal agreements), Dave = statistical correctness (one wrong number). Same anxiety, four separate root causes requiring four separate answers.

- **Validation burden is segment-specific:** Dave's 21 CFR Part 820 means every app update is a potential revalidation event. No other persona mentioned this. In med device, it's a hard stop. Elsewhere, it doesn't exist.

- **Onboarding depth tolerance:** Maria: 4 hours over 2 weeks. Dave: 2 hrs/week, modular. Tameka: 30 minutes or done. Ray: 4 hours total. Same direction (less than vendors expect), wide range on acceptable investment.

- **Community:** Maria and Tameka both asked for industry-segmented user communities unprompted. Dave and Ray did not. Community is real but not universal.

- **DOE:** Dave interested (validates against Minitab). Maria/Tameka/Ray all skipped it. One segment's feature, three segments' noise.

### Strongest Objection

**Dave Kowalski: "One statistically wrong number and I'm done. Permanently."**

Not a pricing objection, not an onboarding objection. A trust threshold with no second chance. In regulated med device, a wrong number has audit consequences. His historical pattern — burned 3 times by quality software — means he arrived pre-skeptical. And this objection travels: one story of a wrong control limit in the med device QE community destroys the segment, not just the account.

### Unmet Needs Discovered

1. **Verbal/informal knowledge capture.** Ray surfaced it most sharply — verbal tolerance agreements between job shop owners and customers. The prompt framed knowledge retention as capturing documented workflows. The personas revealed the real problem is undocumented knowledge in people's heads and informal agreements.

2. **Auditor-ready output generation.** Tameka: CA report that looks like it came from a real QMS. Ray: paper trail for AS9100. Dave: IQ/OQ/PQ validation package. Three regulatory contexts, same need: documents that transfer trust to a third party.

3. **FAI workflow.** Ray: first article inspection tied to part number, CMM output, and objective evidence. Where he loses $6-8K/month. Not mentioned in the prompt or app list.

4. **Supplier nonconformance log.** Tameka: packaging that doesn't meet spec, currently in email. Simple but absent.

### Would Pay

| Persona | Verdict | Price Point | Condition |
|---------|---------|-------------|-----------|
| Maria | **Yes** | $300-400/mo | Automotive reference customer, SPC works on tablet, free trial first |
| Ray | **Yes** | $200-500/mo | ROI proof in 60 days, short-run SPC works, month-to-month option |
| Tameka | **Conditional** | $200-250/mo | Free tier → screenshot for GM → then budget ask. Converts if first result is legible |
| Dave | **Not without structural changes** | $12-18K/yr | IQ/OQ/PQ package, named support contact, Minitab parity on known dataset |

### Language Mining

Phrases the personas used that the prompt did not:

- **"App store for quality tools"** — Maria and Tameka independently. Not prompted. This is how the market names it.
- **"Pay for what we use"** — all four, unprompted.
- **"Keeps knowledge when QEs leave"** — Maria's boss pitch language. Consequence, not feature.
- **"Table stakes"** — Dave on SPC/capability/CAPA.
- **"Show where we give away product"** — Tameka's boss pitch. Economic framing, not quality framing.
- **"Like MeasurLink but..."** — Ray. Comparison-anchored. The "but" is the value prop.
- **"Paper trail for auditors"** — Ray. User's word for "audit trail."
- **"Screenshot-ready"** — Tameka. Output legibility as feature.
- **"Institutional knowledge"** — Dave. Same concept as Maria's "knowledge when QEs leave," different register.
- **"Have you considered reviewing your control plan"** — Ray's example of a generic AI suggestion that triggers cancellation.
- **"Proving ROI in 60 days with specific dollars"** — Ray. Not just ROI — specific dollars, specific timeframe.

### Pricing Model

**Model:** Flat site license, monthly billing, annual option with locked rate. No per-user.

| Segment | Self-approve | Would pay with proof | Comparison anchor |
|---------|-------------|---------------------|-------------------|
| Small food plant (Tameka) | $250/mo | $250/mo | SafetyChain $49/mo |
| Job shop (Ray) | $200/mo | $500/mo (60-day ROI) | MeasurLink (sunk cost) |
| Mid-size auto (Maria) | $300-400/mo | $300-400/mo | Minitab $200/user/mo |
| Enterprise med device (Dave) | Needs VP approval | $12-18K/yr | ETQ $45K + Minitab $18K |

**Free tier is precondition, not differentiator.** Nobody signs before touching real data.

### Onboarding Minimum Viable Experience

1. No demo call before product access (access → result → then human)
2. First meaningful output within one session (chart, screenshot, filled CA — not completed wizard)
3. Upload CSV or connect data source as step one
4. Pre-built templates for first three use cases (replacement for the Excel workbooks being left behind)
5. One human touchpoint, short, on their data — with industry-specific expertise (automotive/FDA/AS9100/SQF)
6. Inspectors/operators require separate track (2-hour video max, tablet-ready)

### Support Tiers

| Price | Response | Channel | Expertise |
|-------|----------|---------|-----------|
| ~$200-250/mo | 24hr email, human reads it | Email + docs + forum | Self-service, industry-segmented community |
| ~$300-500/mo | Same business day | Email + phone/chat | Direct human, no ticket routing |
| ~$1,000+/mo | Same day, <1hr for audit emergencies | Named contact + emergency number | Regulatory-aware (FDA, AS9100, IATF), vendor-provided validation docs |

**Universal failures:** ticket routing, chatbot bouncing, generic responses, 2-day response at premium tier.

The differentiator at higher tiers is not more features — it is faster, more expert humans.
