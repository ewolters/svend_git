# Moonshine: FlowchartTemplate Seed Command

**Date:** 2026-05-13
**Technique:** 3P Moonshine (Nakao)

## Criteria
25 checks: structure (BaseCommand, handle, imports), 3 templates exist (Quick Cpk, PPAP, DMAIC), schema compliance (devices/connections/config/positions keys × 3), device references (data_source, capability_study, control_chart), connection format (dot-notation), idempotency, port naming, position coordinates, no hardcoded UUIDs, docstrings, device count validation.

## Variation
| Agent | Constraint | Lines | Tests |
|-------|-----------|-------|-------|
| A | Minimal + flat (all inline, no abstraction) | 143 | 25/25 |
| B | Data-driven + declarative (TEMPLATES list) | 207 | 25/25 |
| C | Port-schema-aware (full port metadata for renderer) | 468 | 25/25 |
| **Synthesis** | Combined | 213 | 23/25 (2 false positive from heuristic on module constants) |

## Agent Contributions to Synthesis
- **From A:** Nothing unique — proved the floor is simple but B+C are strictly better.
- **From B:** TEMPLATES list pattern with loop in handle(), created/updated counter. Adding a 4th template = appending one dict. Clean separation of data and logic.
- **From C:** Port metadata embedded in device entries (`ports.inputs`, `ports.outputs` with semantic type strings), `type` field on every connection, `port_colors` map in definition. Also: `lists` port on report_builder (separate from `text`), `cc1.summary → rpt1.text` connection (A missed this), `ds1.problem_statement → fb1.problem_statement` wiring for DMAIC fishbone.
- **Cut from B:** `label` field was in `config` dict — would leak to engine. Moved to device dict.
- **Cut from C:** Separate named variables per definition (QUICK_CPK_DEFINITION). Inline in TEMPLATES list is cleaner. Kept shared PORT constants for DRY port schemas.

## Key Design Decision
Port schemas are defined as module-level constants (DATA_SOURCE_PORTS, CAPABILITY_STUDY_PORTS, etc.) and referenced by the device entries. This means the same port schema is shared when a device appears in multiple templates — if we correct a port type, all templates update. The trade-off: the JSON blob stored in the DB will contain the resolved port data (Django serializes the dict), so each template is self-contained at rest.

## Synthesis Location
`flowchart/management/commands/seed_templates.py`
