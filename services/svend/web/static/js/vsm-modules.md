# VSM Module Extraction Plan

Source: `templates/vsm.html` lines 1153-3704 (~2,550 lines inline JS)
Target: `static/js/vsm-*.js` modules

## Module Map

### vsm-state.js (~50 lines)
Global state variables + VSM_ICONS constant.
- Lines 1157-1200: icons, state vars (currentVSM, vsmId, zoom, pan, etc.)
- Undo/redo history

### vsm-api.js (~250 lines)
All API calls wrapped with SvApi.
- `loadVSMList()` — GET /api/vsm/
- `loadVSM(id)` — GET /api/vsm/{id}/
- `saveVSM()` — PUT /api/vsm/{id}/update/
- `createNewVSM()` — POST /api/vsm/
- `addElement(type, x, y)` — POST process-step/inventory/kaizen
- `addMaterialFlow(from, to, type)` — POST process-step/ with connections
- `createFutureState()` — GET /api/vsm/{id}/future-state/
- `compareStates()` — GET /api/vsm/{id}/compare/
- `generateVSMProposals()` — POST /api/vsm/{id}/generate-proposals/
- `approveProposal()` — POST /api/vsm/{id}/approve-proposal/
- `promoteVSM()` — POST /api/hoshin/vsm/{id}/promote/
- `setupProjectSelector()` — GET projects, wire selector

### vsm-canvas.js (~1,000 lines)
SVG rendering — the visual map.
- `renderVSM()` — main render dispatcher
- `renderProcessBox(step, layer)` — process station
- `renderWorkCenter(wc, layer)` — multi-machine station
- `renderInventory(inv, layer)` — triangle/queue/transport
- `renderSupermarket(inv, layer)` — supermarket symbol
- `renderFIFO(inv, layer)` — FIFO lane symbol
- `renderKaizenBurst(burst, layer)` — improvement star
- `renderCustomerSupplier(layer)` — entity boxes
- `renderConnections(layer)` — flow arrows (material + info)
- `renderLeadTimeLadder(layer)` — bottom timeline
- `createSvgIcon()` — SVG icon helper
- Work center helpers: getWorkCenterEffectiveCT, getWorkCenterMembers, associateStepsToWorkCenters

### vsm-interaction.js (~500 lines)
User interaction — drag, select, properties, tools.
- `setupEventListeners()` — canvas click, element select
- `setupDragAndDrop()` — palette → canvas drag
- `showProperties()` / `showInventoryProperties()` / `showKaizenProperties()` / `showEntityProperties()`
- `saveProperties()` — write back to state
- `deleteSelected()` — remove element
- `setTool(tool)` / `setFlowTool(type)` / `handleFlowClick()`
- Canvas transform: updateCanvasTransform, zoomIn, zoomOut, resetView
- Work center resize: startResizeWorkCenter, resizeWorkCenterMove, resizeWorkCenterEnd

### vsm-metrics.js (~300 lines)
Live metrics calculation + step detail panel.
- `updateMetrics()` — recalculate lead time, PCE, bottleneck
- `detectBottleneckClient(vsm)` — find bottleneck station
- `renderSuggestedCalcs(bottleneck)` — context-aware calc links
- `showStepMetrics(step)` — detail overlay with KPIs + annotations
- `renderAnnotationCard()` — calculator result card
- `removeAnnotation()` — delete annotation
- `closeStepMetrics()`
- `formatTime(seconds)` — display helper
- Takt helpers: setTaktDirect, calculateTakt

### vsm-proposals.js (~400 lines)
CI proposals, kaizen, Hoshin integration.
- `loadKaizenHypotheses(burst)` — Synara link
- `createHypothesisFromKaizen()` — create hypothesis
- `loadHypothesisProbabilities()` — fetch priors
- `showTimeline()` — metric history
- `openProposalModal()` / `closeProposalModal()`
- `generateVSMProposals()` — AI proposals
- `renderProposalCards(proposals)` — proposal UI
- `createSelectedProposals()` — batch create
- `approveProposal()` — promote to Hoshin
- `updateProposalButton()` — state indicator
- `exportVSM()` — export functionality

### vsm-list.js (~100 lines)
VSM list/selection view.
- `showVSMList(maps)` — render list view
- `showNewVSMDialog()` / `hideNewVSMDialog()`

## CSS

### static/css/vsm.css (DONE)
All 648 lines extracted from inline <style>.

## Migration Status

- [x] CSS extracted → static/css/vsm.css
- [ ] vsm-state.js
- [ ] vsm-api.js
- [ ] vsm-canvas.js
- [ ] vsm-interaction.js
- [ ] vsm-metrics.js
- [ ] vsm-proposals.js
- [ ] vsm-list.js
- [ ] New vsm-v2.html template using modules
- [ ] Old vsm.html marked deprecated
