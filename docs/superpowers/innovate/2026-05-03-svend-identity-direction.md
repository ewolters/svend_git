# SVEND Identity Direction

**Date:** 2026-05-03
**Status:** Working direction — emerged from red/blue, transfer, morph, and conversation
**Supersedes:** All prior identity framings (tools platform, knowledge platform, problem-solving platform)

## The Identity

**SVEND is an AI-guided operational excellence implementation system.**

The tools do the computation. The AI provides the judgment. The knowledge accumulates automatically.

## The Problem SVEND Is Organized Against

OpEx works. It has a proven track record of breaking records, transforming plants, and creating massive value. But:

1. The systems that drive it are computationally heavy (SPC, DOE, capability) and deeply integrated
2. Few people in the world actually know how to implement them
3. The 99%/1% split: 99% of the market are learners, 1% are veterans who can actually do this

The gap isn't tools. The gap is judgment — knowing what to do next, how to decompose a problem, how to deploy resources, how to ensure execution, how to sustain results.

## Three Levels of Operational Maturity

1. **Vague** — "We had 8% scrap." Can't fix what you can't decompose. Most companies live here.
2. **Structured intelligence** — "124 lb scrap from material A. 89 lb from heavy coating because we use the same nozzle for light and heavy application." Decomposed to the level of causality. Now it's actionable.
3. **Resource deployment** — "Who's working this? How many hours? Who covers their station? How do we ensure it happens and happens right?" Accountability and strategic/tactical thinking about application of resources to problems.

Then **sustaining** is its own layer on top.

SVEND facilitates climbing from level 1 to 3 AND supports execution once you're there.

## How the Pieces Fit

- **AI (Claude)** — The connective tissue. Provides the veteran's judgment: helps decompose vague problems into causal specifics, suggests what tool to use next, structures accountability, remembers what was learned. The AI is the guide that the 99% don't have.
- **PROVA** — Not a user-facing feature. The AI's brain/memory. The knowledge graph builds as a byproduct of doing the work, not as a separate activity. Users never need to understand graph operations.
- **PCL** — What the AI looks at to understand current state. "What's happening on this process?" resolves to characterized measures with provenance and confidence.
- **Analysis Workbench / DSW** — The AI's toolkit. AI dispatches analyses the same way a veteran would walk to a whiteboard and draw a Pareto.
- **Existing tools (SPC, DOE, FMEA, RCA, VSM, QMS, Hoshin)** — The computational backbone. Each tool serves the system; the AI orchestrates when and how they're used.
- **Tutorial/onboarding** — Not a walkthrough. It's the first conversation. "What's your biggest problem right now? Let's look at it together."

## What This Changes

- **PROVA** doesn't need a frontend or hypothesis builder UI. It needs to be a memory layer the AI reads/writes during problem-solving conversations.
- **PCL** is the AI's view of reality, not data infrastructure for its own sake.
- **The workbench** is the AI's toolkit, not the product itself.
- **Identity isn't expressed through UX patterns or named journeys.** It's expressed through the quality of the conversation with the AI — the AI IS the product experience.
- **The "in between" that the 99% lack** — the judgment calls, the "what do I look at next," the decomposition from vague to causal — is delivered through AI guidance, not through better UI or workflow automation.

## Why Now

The world is moving toward AI. Two years ago, encoding veteran judgment meant building a knowledge graph users would manually populate. Now, the AI can provide that judgment in real-time and build the knowledge graph as a byproduct. The timing is right for this identity in a way it wasn't before.

## Origin

This direction emerged from:
1. Red/blue on PROVA — exposed that PROVA is sound but premature as a user-facing feature
2. Cross-domain transfer on identity — four domains said "the connective tissue IS the identity"
3. Morphological analysis — showed the differentiator is how the system talks to users, not the tools themselves
4. Direct conversation — Eric's operating experience: the "in between" where veterans make judgment calls is where OpEx succeeds or fails, and that's what the 99% lack

Artifacts:
- `2026-05-03-prova-redblue.md`
- `2026-05-03-svend-identity-transfer.md`
- `2026-05-03-svend-identity-morph.md`
