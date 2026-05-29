# Moonshine: VSM→Hoshin Flowchart Integration

**Date:** 2026-05-14
**Technique:** 3P Moonshine (Nakao)

## Criteria
10 structural checks: device class definition, charter output, Hoshin connection, PluginOutput usage, financial analysis, no direct ORM in devices, semantic port types, flowchart system integration, VSM diff concept, structured actionable output.

## Variation
| Agent | Constraint | Lines | Tests |
|-------|-----------|-------|-------|
| A | Thin Bridge — one device + bus subscriber | 247 | 9/10 |
| B | Hoshin IS Flowchart — cascade + x_matrix as devices | 290 | 10/10 |
| C | Contract Envelope — universal interface, any source→any sink | 259 | 10/10 |
| **Synthesis** | C's contract + B's Hoshin devices | 689 (4 files) | 15/15 |

## Agent Contributions to Synthesis
- From A: Confirmed charter generation is just a contract_envelope configured for VSM — no dedicated charter device needed
- From B: strategic_cascade and x_matrix devices — the Hoshin-specific computation layer
- From C: **The key insight** — contract_envelope as universal interface. Any methodology produces contracts. Any tracker consumes them. The router decides where they go.

## Why C Won
The contract envelope scales. Agent A's bridge was VSM→Hoshin only. Adding FMEA→Hoshin means another bridge. Agent C's contract is the interface boundary — VSM, FMEA, capability, A3, RCA all produce the same contract. One router handles all destinations.

Agent B's Hoshin devices were essential but Hoshin-specific. Combined with C's universal contract, you get: any analysis → contract_envelope → contract_router → strategic_cascade → x_matrix.

## Synthesis Location
- `plugins/contract_envelope.py` — universal contract packager
- `plugins/contract_router.py` — priority-based routing (strategic/tactical/quick_win)
- `plugins/strategic_cascade.py` — Hoshin objective → breakthrough projects
- `plugins/x_matrix.py` — cross-reference matrix
- Commit: 23337f7
