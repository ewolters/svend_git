---
area: architecture
---

# Forge Package Independence

## Required Patterns

- Forge packages (forgespc, forgestat, forgeviz, forgenarr) are independent of kjerne
- forgespc produces result objects, forgeviz converts them to ChartSpec
- New result types get new converters in forgeviz (from_X_result pattern)
- analysis/handlers/ call forge packages directly, never legacy agents_api code
- Narrative generation goes through forgenarr, not inline in handlers

## Anti-Patterns

- Importing from agents_api/ inside analysis/handlers/ (creates legacy dependency)
- Using _plotly_to_chartspec() as default strategy (acceptable ONLY for pending native ports)
- Writing computation logic in handlers instead of forge packages
- Duplicating forgespc math in kjerne when the package already has the function
