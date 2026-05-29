# Post-Mortem: Session Process — Workstream → Lab → Moonshine → FTB

**Date:** 2026-05-13
**Technique:** Retrospective (adapted pre-mortem — 3 isolated perspectives on what worked)
**Perspectives:** Tool Builder, Next Session, Lean Practitioner

## Tool Builder Perspective

The session worked better than expected, but not because the tools were designed to compose — they composed because the workflow happened to be linear. Each tool produced an artifact that the next consumed, and that's the happy path. Workstream-to-Lab was seamless: 60 decisions front-loaded, Lab immediately knew what to explore. Lab-to-Moonshine was clean because the notebook was a self-contained snapshot — exactly what parallel agents need. FTB after Moonshine was the awkward joint: Claude manually read the synthesis and decided where to place 31 tags. No handoff protocol between Moonshine synthesis and FTB tagging. The quality depends entirely on Claude's comprehension.

Context savings: ~15-20K tokens of read/explore overhead avoided. Lab saved 16 tool calls. Workstream saved 10-15 min of exploration. Moonshine agents ran in background (separate context windows).

This is a prototyping assembly line, not a general one. Bug fixes, refactors, and incremental work would skip Moonshine and probably Lab. That's fine — prototyping is where the most context gets wasted.

## Next Session Perspective

The workstream JSON was stale after the session — `current_state` and `next_steps` didn't reflect completed work (since fixed). Ephemeral `/tmp/` prototypes are gone but all durable artifacts survived: the committed management command, the FTB spec, the moonshine doc. 31 FTB tags in the code will help if the FTB plugin is loaded and the spec doc exists for context.

The real gap: next steps must be specific actions ("wire template picker modal"), not categories ("build UI"). Specificity is what lets the next session start working in five minutes instead of thirty.

## Lean Practitioner Perspective

Context window IS takt time. Every token consumed is capacity spent; when the window fills, the line stops. Workstream load is SMED — sixty decisions pre-loaded means zero changeover time. Lab batching is a milk run — one planned route instead of seventeen trips. The pull signal is the workstream JSON itself: it defines done, and each step pulls the next.

Moonshine preserves the essential mechanism of 3P: multiple competing approaches tested against objective criteria before commitment. What's preserved: the discipline. What's lost: the team. Three agents from the same model explore a narrower solution space than three engineers with different backgrounds. Sufficient for well-defined problems; insufficient for novel architecture.

FTB improves on TWI Job Instruction: documentation cannot drift from work because it IS the work. What it loses is teachability — specs are machine-readable but not optimized for cold onboarding.

The single question ("self-contained JSON?") that locked three decisions is expert dependency, not respect for people. A model cell must not require a brilliant question to function. The process surfaced the question (FTB tagged it), but answering it required domain expertise the process can't provide.

"Where is your problem?" A session that runs this smoothly either has no problems or has problems you cannot see yet.

## Pattern Analysis

**Common themes:** (1) Workstream load is highest-leverage tool — all three agreed. (2) Claude as integration bus is feature and risk — tools compose through Claude's comprehension, not directly.

**Unique risks:** (1) Moonshine agent diversity narrows vs human teams. (2) No cycle time measurement — "a model cell without metrics is a demo." (3) Expert dependency on the critical question.

**Earliest warning sign:** A session without the right human intervention at the right moment produces technically correct code on a wrong foundation.

**What to codify:**

| Keep | Fix | Watch |
|------|-----|-------|
| Workstream load as session start | Always update workstream before session ends | Moonshine diversity on novel problems |
| Lab for 3+ command investigations | Moonshine→FTB handoff manifest | Expert dependency |
| FTB tag-as-you-go for prototypes | Specific next steps, not categories | Context pressure on larger features |
| Moonshine for multi-approach features | Add cycle time measurement | |
