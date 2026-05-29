# Pressure Test: Game-Style Keybinding/Hotbar Navigation

**Date:** 2026-05-05
**Techniques:** Focus Group (4 personas) + S1/S2 Conference
**Subject:** Video-game-style hotbar/keybinding navigation for SVEND

---

## Focus Group Results

### Consensus (all four agreed)

1. **"No menus" is universally validated.** Every persona appreciated bypassing menu navigation. Different framings, same endorsement.
2. **Data ingestion is the real barrier.** Before any hotbar matters, data has to get in. Merged cells, silent row drops, revision tracking. The 10 minutes BEFORE the analysis is the hard problem.
3. **"Video game" terminology kills before the demo starts.** Dave almost closed the tab. Destiny assumed it wasn't for her. Only Maria (already champion) looked past it.

### What Landed

| Feature | Maria | Dave | Kenji | Destiny |
|---|---|---|---|---|
| Hotbar (1-2-3-4) | "Mirrors actual workflow" | "Faster than any dropdown" | "That's my Monday chain" | "Thumb goes there anyway" |
| Q (quick run) | — | "Almost sells me by itself" | — | — |
| M (map) | "Selling point IF interrogatable" | — | "Replaces 15-min Director explanation" | — |
| Templates | "Karen memorizes in a week" | "Need to rearrange MY way" | "Use once, modify, save own" | Doesn't apply |
| Inventory drag | "Show me the audit trail of the bind" | "The drag is fun; import is hard" | "Show me what happened after the drop" | "What dataset? I don't have datasets." |

### What Broke

| Problem | Who | Severity |
|---|---|---|
| "Video game" framing | Dave, Kenji, Destiny | **Kill-before-demo.** Retire from all customer-facing language. |
| Hints fade after 3 uses | Destiny | High. Off 4 days between shifts. Needs persistent "?" button. |
| "Drag dataset" assumes data literacy | Destiny | High. She doesn't have "datasets" — she has measurements. |
| Multi-engineer collaboration / state | Maria | Medium-high. Two people in same project = who has state? |
| Shared datasets across projects | Kenji | Medium. Same data, different chains = version control. |
| AS9100 traceability of Q outputs | Dave | Medium. Quick runs must be recoverable for audit. |
| Messy Excel (merged cells, notes) | Dave | Medium. Data format rigidity kills day one. |
| Spec limits required before chart | Dave | Medium. "Give me chart first, add specs after." |
| Hotbar overflow past 4-5 tools | Conference | Design gap. What's the convention for tool 10? |
| iPad at 12% battery, cracked screen | Destiny | Physical reality. Can't design around it, but must acknowledge. |

### Strongest Objection

**Dave's 20-minute test:** "Fastest way from raw data to customer-ready capability report. 18 minutes. No training." If Q (quick run) requires ANY setup — wizard, format specification, account creation — it fails. The tool is being evaluated on its smallest mode. If that mode works, the chain features become relevant. If not, nothing else matters.

### Unmet Need: Graceful Re-Entry

Destiny needs the tool to remember context across multi-day gaps. Dave needs half-built analyses to persist across 40 context-switches/day. The tool implicitly rewards frequency of use and punishes gaps. Needs: persistent help, resumable state, re-orientation on return.

### Language Mining

**Use:**
- "Keyboard shortcuts instead of menus" (Maria)
- "Muscle memory" (Dave)
- "One-shot" (via Q)
- "When data changes we're not rebuilding" (Kenji)
- "Customer-ready" (Dave)
- "Pass/fail" (Destiny)
- "Chain" (Kenji)

**Retire:**
- "Video game" — 3/4 negative before engaging
- "Inventory" as UI label — respondents hear "warehouse" not "library"
- "Data bus" — never use externally

**Test next:** "The fastest path from raw data to a report your customer will accept."

---

## S1/S2 Conference Results

### Agreements (both sides)

1. Hotbar/keybinding correct for daily power user. Muscle memory > menus.
2. Template-as-entry-point correct. "Borrow and break" validated.
3. Traditional menus wrong for 30+ tools. Problem is real.
4. Drag-from-inventory directionally right but incomplete. Multi-dataset, column mapping, data quality need solutions.
5. Enterprise perception problem exists and is unresolved.

### Disagreements

| Topic | S1 (Innovator) | S2 (Conservative) | Crux |
|---|---|---|---|
| Occasional/monthly user | Not primary target; templates help | Removal of menus = no fallback for infrequent users | Is "occasional user" a target or acceptable casualty? |
| Enterprise perception | Presentation problem (add professional skin) | Architectural signal problem (game metaphor is load-bearing) | Can professional mode be bolted on, or does the metaphor commit the surface? |
| Keybinding completeness | Starting point; expansion solvable | Hotbar overflow + discovery + accessibility = architecture, not details | Is this MVP that expands cleanly, or proof of concept needing more design? |

### Open Questions for Founder

1. **Primary buyer vs. primary user?** IT evaluator (demo matters) vs. bottom-up individual buyer (power UX is everything)?
2. **Is occasional-use support a business requirement?** Annual PPAP engineer — target or not?
3. **Discovery path for tools 5-15?** Unanswered. Clean solution validates model; if it requires new nav surface, model changes.
4. **"Game UI" — feature or liability for aerospace/automotive/medical?** Market-specific. Founder knows buyer.
5. **Column mapping and multi-dataset tools?** Design gap. If answer requires config layer, changes "drag replaces everything" claim.
6. **Accessibility compliance?** Keyboard-native model is well-positioned for a11y if designed for it from start.

### What Is Resolved

Direction correct. Project-based, hotbar for frequent tools, template entry, drag-from-inventory as primary data routing. Better than status quo for primary cohort.

### What Is Not Resolved

Model incomplete as spec. Gaps: occasional users, enterprise perception, tool discovery beyond 4, accessibility, multi-dataset drag, graceful re-entry. These are design problems, not blocking objections — but they need resolution before implementation commitment.

---

## Combined Verdict

**The interaction model works. The framing doesn't. The gaps are specific and solvable.**

### Confirmed:
- Hotbar/keybinding as primary navigation ✓
- Q (quick run) as one-shot mode ✓  
- M (map) as on-demand project view ✓
- Templates as entry point ✓
- Drag-to-bind as data routing (with caveats) ✓
- Full-screen tool with minimal chrome ✓

### Must Fix Before Shipping:
1. **Kill "video game" from all language.** Say "keyboard shortcuts" or "muscle memory navigation."
2. **Q must work in <2 minutes.** Paste data → chart → Cpk. No wizard, no spec-limits-first gate.
3. **Drag-to-bind needs audit trail.** Every binding recorded, visible in Map, exportable.
4. **Data ingestion feedback.** After drop: row count, column preview, quality flags. No silent failures.
5. **Hints must be recallable.** Persistent "?" or long-press-for-help. Don't punish gaps between shifts.
6. **Discovery for tools beyond the hotbar.** What's the path to tool 8 that isn't on your hotbar?
7. **Graceful re-entry.** Resumable state. "You were here" on return.
8. **Column mapping UX.** Drag gets you 80% there; the remaining 20% (which col is measurement vs. grouping?) needs a solution that isn't a config wizard.

### Open Design Questions:
- Multi-engineer collaboration (who has state?)
- Shared datasets across projects (version control)
- Export/reporting for non-users (Director, SQE, auditor)
- Accessibility / screen reader compliance
- Enterprise demo mode vs. daily-use mode
