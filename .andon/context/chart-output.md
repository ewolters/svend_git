---
area: visualization
---

# Chart Output Patterns

## Required Patterns

- All chart output MUST be ForgeViz ChartSpec (from forgeviz.core.spec)
- Use from_spc_result(), from_conformal_result(), from_mewma_result() for SPC
- Return charts in "charts" key from handlers, chain.assemble() serializes to "plots"
- ForgeViz builders live in forgeviz.charts.* — use the appropriate module

## Anti-Patterns

- Raw Plotly trace dicts ({"type": "scatter", "data": [...]})
- Importing plotly or referencing Plotly layout/trace builders
- Building chart JSON inline instead of using forgeviz builders
- Returning "plots" key from handlers (use "charts" — chain handles the rename)
