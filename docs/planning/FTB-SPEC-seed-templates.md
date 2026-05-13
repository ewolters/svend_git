# FTB Specification

Compiled: 2026-05-13T08:56:18.182237
Source: `/home/eric/kjerne/flowchart/management/commands/seed_templates.py`
Total tags: 32

## Contents

- [Specification Notes](#spec) (7)
- [API Surface](#api) (1)
- [Wiring & Integration](#wire) (4)
- [Decisions](#decision) (6)
- [TODO Items](#todo) (11)
- [Open Questions](#question) (2)
- [Performance Notes](#perf) (1)

## Locked Decisions

- **Management command, not data migration. Templates are** (`flowchart/management/commands/seed_templates.py:9`)
- **PORT_COLORS lives in template definition JSON, not in a** (`flowchart/management/commands/seed_templates.py:25`)
- **lists port is SEPARATE from text port. Fishbone causes** (`flowchart/management/commands/seed_templates.py:146`)
- **TEMPLATES list is the sole data source. Adding a 4th** (`flowchart/management/commands/seed_templates.py:163`)

## Open Questions

- Should port schemas be derived from Plugin classes instead? (`flowchart/management/commands/seed_templates.py:47`)
- Should data_source be a single plugin with dynamic ports, (`flowchart/management/commands/seed_templates.py:60`)

---

## Specification Notes
_Core behavior and requirements extracted from code_

### `flowchart/management/commands/seed_templates.py`

- L28: 8 categories map 1:1 to the semantic type system (types.py).
- L92: target port intentionally omitted from templates. Target is optional
- L109: subgroup_size is a config port. When NOT connected, renderer shows
- L143: All report_builder input ports use wildcard types (chart:*, metric:*).
- L205: PPAP is Dana's use case. Must produce customer-ready output.
- L256: DMAIC is Carmen's use case. 50-70 students/yr load this template.
- L320: Idempotent via update_or_create keyed on name. Running twice

## API Surface
_Endpoints, interfaces, and contracts_

### `flowchart/management/commands/seed_templates.py`

- L165 **[contract]**: Template definition JSON schema (consumed by renderer):

## Wiring & Integration
_How components connect_

### `flowchart/management/commands/seed_templates.py`

- L12 **[output]**: Writes to flowchart_template table. Consumed by:
- L88 **[input]**: capability_study plugin EXISTS (plugins/capability.py).
- L108 **[input]**: control_chart plugin EXISTS (plugins/control_chart.py).
- L259 **[output]**: When templates are shared, they appear in all users'

## Decisions
_Architecture and design decisions_

### `flowchart/management/commands/seed_templates.py`

- L9 **[locked]**: Management command, not data migration. Templates are
- L25 **[locked]**: PORT_COLORS lives in template definition JSON, not in a
- L43 **[tentative]**: Port schemas are defined HERE (in seed data), not
- L73 **[tentative]**: DMAIC data_source has extra problem_statement port.
- L146 **[locked]**: lists port is SEPARATE from text port. Fishbone causes
- L163 **[locked]**: TEMPLATES list is the sole data source. Adding a 4th

## TODO Items
_Outstanding work_

### `flowchart/management/commands/seed_templates.py`

- L14 **[P0]**: data_source, report_builder, fishbone plugins DO NOT EXIST YET.
- L17 **[P1]**: Tenant scoping. Seed templates have no tenant_id (system-wide).
- L52 **[P1]**: Write a test that loads each template, finds its plugins in
- L56 **[P0]**: data_source plugin does not exist. When built, it must:
- L124 **[P0]**: fishbone plugin does not exist. When built:
- L138 **[P0]**: report_builder plugin does not exist. This is the most
- L169 **[P1]**: Template versioning. When we update a template definition,
- L173 **[P2]**: Template categories/tags for the picker UI. Quick Cpk = "getting
- L261 **[P2]**: "Green Belt" is ILSSI branding. May need to parameterize
- L322 **[P1]**: Add --dry-run flag to show what WOULD change without writing DB.
- L323 **[P2]**: Add --validate flag to check all referenced plugins exist

## Open Questions
_Unresolved design questions_

### `flowchart/management/commands/seed_templates.py`

- L47: Should port schemas be derived from Plugin classes instead?
- L60: Should data_source be a single plugin with dynamic ports,

## Performance Notes
_Performance considerations and constraints_

### `flowchart/management/commands/seed_templates.py`

- L208: 4 devices, 10 connections. Execution is sequential (topo sort).
