/**
 * VSM Interaction Module
 * Drag/drop, properties, tools, zoom/pan, flow drawing.
 * MIGRATION: Extracted from templates/vsm.html
 */


async function addElement(type, x, y, delayType = '') {
    saveVSMState();
    if (!vsmId) {
        // Need to create VSM first
        showNewVSMDialog();
        return;
    }

    let endpoint = '';
    let data = { x, y };

    switch(type) {
        case 'process':
            endpoint = `/api/vsm/${vsmId}/process-step/`;
            data.name = 'Process';
            data.cycle_time = 30;
            data.uptime = 95;
            data.operators = 1;
            break;
        case 'inventory':
            endpoint = `/api/vsm/${vsmId}/inventory/`;
            data.days_of_supply = 1;
            data.delay_type = delayType || 'inventory';
            break;
        case 'supermarket':
            // Supermarket is a special inventory type
            endpoint = `/api/vsm/${vsmId}/inventory/`;
            data.days_of_supply = 0.5;
            data.delay_type = 'supermarket';
            data.is_supermarket = true;
            break;
        case 'fifo':
            // FIFO lane is a buffer element stored as inventory
            endpoint = `/api/vsm/${vsmId}/inventory/`;
            data.days_of_supply = 0;
            data.delay_type = 'fifo';
            data.is_fifo = true;
            data.max_quantity = 10;
            break;
        case 'kaizen':
            endpoint = `/api/vsm/${vsmId}/kaizen/`;
            data.text = 'Improvement';
            data.priority = 'medium';
            break;
        case 'workcenter': {
            // Stored client-side, saved via saveVSM()
            saveVSMState();
            if (!currentVSM.work_centers) currentVSM.work_centers = [];
            currentVSM.work_centers.push({
                id: 'wc-' + Math.random().toString(36).substr(2, 8),
                name: 'Work Center',
                x, y,
                width: 280,
                height: 200
            });
            associateStepsToWorkCenters();
            renderVSM();
            await saveVSM();
            return;
        }
        case 'customer':
        case 'supplier': {
            // Stored client-side in customers/suppliers arrays, saved via saveVSM()
            saveVSMState();
            const arr = type === 'customer' ? 'customers' : 'suppliers';
            if (!currentVSM[arr]) currentVSM[arr] = [];
            currentVSM[arr].push({
                id: Math.random().toString(36).substr(2, 8),
                name: type === 'customer' ? 'Customer' : 'Supplier',
                detail: '',
                x, y
            });
            renderVSM();
            await saveVSM();
            return;
        }
        default:
            return;
    }

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Failed to add element');
        const result = await response.json();
        currentVSM = result.vsm;
        renderVSM();
        updateMetrics();
    } catch (err) {
        console.error('Add element error:', err);
    }
}

let dragElement = null;
let dragOffsetX = 0;
let dragOffsetY = 0;

let wasDragged = false;


async function addMaterialFlow(fromId, toId, flowType) {
    if (!currentVSM) return;
    saveVSMState();

    // Initialize material_flow if needed
    if (!currentVSM.material_flow) {
        currentVSM.material_flow = [];
    }

    // Add connection
    currentVSM.material_flow.push({
        id: Math.random().toString(36).substr(2, 8),
        from_step_id: fromId,
        to_step_id: toId,
        type: flowType
    });

    renderVSM();
    await saveVSM();
}

// =============================================================================
// Dialogs
// =============================================================================

function deleteSelected() {
    if (!selectedElement || !currentVSM) return;
    saveVSMState();

    // Find and remove from appropriate array
    const id = selectedElement.id;
    currentVSM.process_steps = (currentVSM.process_steps || []).filter(s => s.id !== id);
    currentVSM.inventory = (currentVSM.inventory || []).filter(i => i.id !== id);
    currentVSM.kaizen_bursts = (currentVSM.kaizen_bursts || []).filter(k => k.id !== id);
    currentVSM.customers = (currentVSM.customers || []).filter(c => c.id !== id);
    currentVSM.suppliers = (currentVSM.suppliers || []).filter(s => s.id !== id);
    currentVSM.work_centers = (currentVSM.work_centers || []).filter(w => w.id !== id);
    // If a work center was deleted, clear work_center_id from its members
    (currentVSM.process_steps || []).forEach(step => {
        if (step.work_center_id === id) delete step.work_center_id;
    });

    selectedElement = null;
    renderVSM();
    updateMetrics();
    saveVSM();
}

// =============================================================================
// Metrics & Actions
// =============================================================================

function dragElementEnd() {
    if (dragElement) {
        associateStepsToWorkCenters();
        renderVSM();
        saveVSM();
        dragElement = null;
    }
    document.removeEventListener('mousemove', dragElementMove);
    document.removeEventListener('mouseup', dragElementEnd);
}


function dragElementMove(e) {
    if (!dragElement) return;
    wasDragged = true;

    const canvasRect = document.getElementById('canvas-container').getBoundingClientRect();
    const x = (e.clientX - canvasRect.left - dragOffsetX - panX) / zoom;
    const y = (e.clientY - canvasRect.top - dragOffsetY - panY) / zoom;

    dragElement.x = x;
    dragElement.y = y;
    renderVSM();
}


function handleFlowClick(step) {
    if (currentTool !== 'flow' || !currentFlowType) return;

    if (!flowSourceStep) {
        // First click - select source
        flowSourceStep = step;
        document.getElementById('flow-hint').textContent =
            `Source: "${step.name || 'Process'}". Now click the destination box.`;
        // Highlight source element on canvas
        const el = document.querySelector(`[data-id="${step.id}"]`);
        if (el) el.classList.add('flow-source-highlight');
    } else if (flowSourceStep.id !== step.id) {
        // Second click - create connection
        addMaterialFlow(flowSourceStep.id, step.id, currentFlowType);
        // Remove highlight
        document.querySelectorAll('.flow-source-highlight').forEach(
            e => e.classList.remove('flow-source-highlight'));
        flowSourceStep = null;
        document.getElementById('flow-hint').textContent =
            'Flow created! Click another source to add more, or switch tools.';
    }
}


function resetView() {
    zoom = 1;
    panX = 0;
    panY = 0;
    updateCanvasTransform();
}


function resizeWorkCenterEnd() {
    if (!resizingWC) return;
    associateStepsToWorkCenters();
    renderVSM();
    saveVSM();
    resizingWC = null;
    document.removeEventListener('mousemove', resizeWorkCenterMove);
    document.removeEventListener('mouseup', resizeWorkCenterEnd);
}


function resizeWorkCenterMove(e) {
    if (!resizingWC) return;
    const dx = (e.clientX - resizeStartX) / zoom;
    const dy = (e.clientY - resizeStartY) / zoom;
    resizingWC.width = Math.max(160, resizeStartW + dx);
    resizingWC.height = Math.max(100, resizeStartH + dy);
    renderVSM();
}


function saveProperties() {
    if (!selectedElement || !currentVSM) return;
    saveVSMState();

    if (selectedElementType === 'workcenter') {
        selectedElement.name = document.getElementById('prop-name').value;
        selectedElement.width = parseInt(document.getElementById('prop-wc-width').value) || 280;
        selectedElement.height = parseInt(document.getElementById('prop-wc-height').value) || 200;
        associateStepsToWorkCenters();
    } else if (selectedElementType === 'kaizen') {
        selectedElement.text = document.getElementById('prop-kaizen-text').value;
        selectedElement.priority = document.getElementById('prop-kaizen-priority').value;
        const hId = document.getElementById('prop-kaizen-hypothesis').value;
        selectedElement.hypothesis_id = hId || null;
    } else if (selectedElementType === 'customer' || selectedElementType === 'supplier') {
        selectedElement.name = document.getElementById('prop-name').value;
        selectedElement.detail = document.getElementById('prop-entity-detail').value;
    } else if (selectedElementType === 'inventory') {
        // Save inventory properties
        selectedElement.name = document.getElementById('prop-name').value;
        selectedElement.delay_type = document.getElementById('prop-delay-type').value;
        selectedElement.days_of_supply = parseFloat(document.getElementById('prop-dos').value) || 0;

        // Update special flags based on delay type
        selectedElement.is_supermarket = selectedElement.delay_type === 'supermarket';
        selectedElement.is_fifo = selectedElement.delay_type === 'fifo';
    } else {
        // Save process properties
        selectedElement.name = document.getElementById('prop-name').value;
        selectedElement.cycle_time = parseFloat(document.getElementById('prop-cycle-time').value) || null;
        selectedElement.changeover_time = parseFloat(document.getElementById('prop-changeover').value) || null;
        selectedElement.uptime = parseFloat(document.getElementById('prop-uptime').value) || null;
        selectedElement.operators = parseInt(document.getElementById('prop-operators').value) || null;
        selectedElement.batch_size = parseInt(document.getElementById('prop-batch').value) || null;
        selectedElement.scrap_rate = parseFloat(document.getElementById('prop-scrap').value) || null;
        selectedElement.available_time = parseInt(document.getElementById('prop-available').value) || null;
        selectedElement.shifts = parseInt(document.getElementById('prop-shifts').value) || null;
    }

    document.getElementById('properties-panel').classList.remove('visible');
    renderVSM();
    updateMetrics();
    saveVSM();
}


function setFlowTool(flowType) {
    currentTool = 'flow';
    currentFlowType = flowType;
    flowSourceStep = null;
    document.querySelectorAll('.toolbar-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.flow-item').forEach(f => f.classList.remove('active'));
    const flowItem = document.querySelector(`.flow-item[data-flow="${flowType}"]`);
    if (flowItem) flowItem.classList.add('active');
    document.getElementById('flow-hint').textContent =
        `${flowType.charAt(0).toUpperCase() + flowType.slice(1)} mode: click a source process box.`;
}


function setTool(tool) {
    currentTool = tool;
    currentFlowType = null;
    flowSourceStep = null;
    document.querySelectorAll('.toolbar-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.flow-item').forEach(f => f.classList.remove('active'));
    const btn = document.querySelector(`.toolbar-btn[data-tool="${tool}"]`);
    if (btn) btn.classList.add('active');
}


function setupDragAndDrop() {
    const paletteItems = document.querySelectorAll('.palette-item[draggable="true"]');
    const canvasContainer = document.getElementById('canvas-container');

    paletteItems.forEach(item => {
        item.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('element-type', item.dataset.type);
            e.dataTransfer.setData('delay-type', item.dataset.delay || '');
            item.classList.add('dragging');
        });

        item.addEventListener('dragend', () => {
            item.classList.remove('dragging');
        });
    });

    canvasContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
    });

    canvasContainer.addEventListener('drop', async (e) => {
        e.preventDefault();
        const type = e.dataTransfer.getData('element-type');
        const delayType = e.dataTransfer.getData('delay-type');
        if (!type) return;

        const rect = canvasContainer.getBoundingClientRect();
        const x = (e.clientX - rect.left - panX) / zoom;
        const y = (e.clientY - rect.top - panY) / zoom;

        await addElement(type, x, y, delayType);
    });
}

// =============================================================================
function setupEventListeners() {
    // Tool buttons
    document.querySelectorAll('.toolbar-btn[data-tool]').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.toolbar-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentTool = btn.dataset.tool;
        });
    });

    // Canvas interactions
    const canvas = document.getElementById('vsm-canvas');

    canvas.addEventListener('mousedown', (e) => {
        if (currentTool === 'pan' || e.button === 1) {
            isPanning = true;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
            canvas.style.cursor = 'grabbing';
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        if (isPanning) {
            const dx = e.clientX - lastMouseX;
            const dy = e.clientY - lastMouseY;
            panX += dx;
            panY += dy;
            updateCanvasTransform();
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        }
    });

    canvas.addEventListener('mouseup', () => {
        isPanning = false;
        canvas.style.cursor = currentTool === 'pan' ? 'grab' : 'default';
    });

    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        zoom = Math.max(0.25, Math.min(4, zoom * delta));
        updateCanvasTransform();
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        // Ctrl+Z / Ctrl+Shift+Z for undo/redo
        if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
            e.preventDefault();
            undoVSM();
            return;
        }
        if ((e.ctrlKey || e.metaKey) && (e.key === 'Z' || (e.key === 'z' && e.shiftKey))) {
            e.preventDefault();
            redoVSM();
            return;
        }
        if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
            e.preventDefault();
            redoVSM();
            return;
        }

        switch(e.key.toLowerCase()) {
            case 'v':
                setTool('select');
                break;
            case 'm':
                setTool('material-flow');
                break;
            case 'i':
                setTool('info-flow');
                break;
            case 'delete':
            case 'backspace':
                deleteSelected();
                break;
        }
    });
}


function showEntityProperties(entity, entityType) {
    const panel = document.getElementById('properties-panel');
    panel.classList.add('visible');
    selectedElementType = entityType; // 'customer' or 'supplier'

    const isCustomer = entityType === 'customer';
    document.getElementById('prop-panel-title').textContent = isCustomer ? 'Customer' : 'Supplier';
    // Hide all non-relevant groups
    document.getElementById('prop-delay-group').style.display = 'none';
    document.getElementById('prop-dos-group').style.display = 'none';
    document.getElementById('prop-kaizen-text-group').style.display = 'none';
    document.getElementById('prop-kaizen-priority-group').style.display = 'none';
    document.getElementById('prop-kaizen-hypothesis-group').style.display = 'none';
    document.getElementById('prop-wc-group').style.display = 'none';
    document.querySelectorAll('.prop-row').forEach(r => r.style.display = 'none');
    // Show entity detail field
    document.getElementById('prop-entity-detail-group').style.display = 'block';
    document.getElementById('prop-entity-detail-label').textContent = isCustomer ? 'Demand' : 'Supply Frequency';
    document.getElementById('prop-name').value = entity.name || (isCustomer ? 'Customer' : 'Supplier');
    document.getElementById('prop-entity-detail').value = entity.detail || '';

    selectedElement = entity;
}


function showInventoryProperties(inv) {
    const panel = document.getElementById('properties-panel');
    panel.classList.add('visible');
    selectedElementType = 'inventory';

    // Show inventory fields, hide all others
    document.getElementById('prop-panel-title').textContent = 'Delay/Buffer Properties';
    document.getElementById('prop-delay-group').style.display = 'block';
    document.getElementById('prop-dos-group').style.display = 'block';
    document.getElementById('prop-kaizen-text-group').style.display = 'none';
    document.getElementById('prop-kaizen-priority-group').style.display = 'none';
    document.getElementById('prop-kaizen-hypothesis-group').style.display = 'none';
    document.getElementById('prop-entity-detail-group').style.display = 'none';
    document.getElementById('prop-wc-group').style.display = 'none';
    document.querySelectorAll('.prop-row').forEach(r => r.style.display = 'none');

    document.getElementById('prop-name').value = inv.name || '';
    document.getElementById('prop-delay-type').value = inv.delay_type || 'inventory';
    document.getElementById('prop-dos').value = inv.days_of_supply || '';

    selectedElement = inv;
}


function showKaizenProperties(burst) {
    const panel = document.getElementById('properties-panel');
    panel.classList.add('visible');
    selectedElementType = 'kaizen';

    document.getElementById('prop-panel-title').textContent = 'Kaizen Burst';
    // Hide all field groups
    document.getElementById('prop-delay-group').style.display = 'none';
    document.getElementById('prop-dos-group').style.display = 'none';
    document.getElementById('prop-entity-detail-group').style.display = 'none';
    document.getElementById('prop-wc-group').style.display = 'none';
    document.querySelectorAll('.prop-row').forEach(r => r.style.display = 'none');
    // Show kaizen fields
    document.getElementById('prop-kaizen-text-group').style.display = 'block';
    document.getElementById('prop-kaizen-priority-group').style.display = 'block';

    document.getElementById('prop-name').value = burst.text || '';
    document.getElementById('prop-kaizen-text').value = burst.text || '';
    document.getElementById('prop-kaizen-priority').value = burst.priority || 'medium';

    // Hypothesis linking
    document.getElementById('prop-kaizen-hypothesis-group').style.display = 'block';
    loadKaizenHypotheses(burst);

    selectedElement = burst;
}


function showProperties(element) {
    const panel = document.getElementById('properties-panel');
    panel.classList.add('visible');
    selectedElementType = 'process';

    // Show process fields, hide all others
    document.getElementById('prop-panel-title').textContent = 'Process Properties';
    document.getElementById('prop-delay-group').style.display = 'none';
    document.getElementById('prop-dos-group').style.display = 'none';
    document.getElementById('prop-kaizen-text-group').style.display = 'none';
    document.getElementById('prop-kaizen-priority-group').style.display = 'none';
    document.getElementById('prop-kaizen-hypothesis-group').style.display = 'none';
    document.getElementById('prop-entity-detail-group').style.display = 'none';
    document.getElementById('prop-wc-group').style.display = 'none';
    document.querySelectorAll('.prop-row').forEach(r => r.style.display = 'flex');

    document.getElementById('prop-name').value = element.name || '';
    document.getElementById('prop-cycle-time').value = element.cycle_time || '';
    document.getElementById('prop-changeover').value = element.changeover_time || '';
    document.getElementById('prop-uptime').value = element.uptime || '';
    document.getElementById('prop-operators').value = element.operators || '';
    document.getElementById('prop-batch').value = element.batch_size || '';
    document.getElementById('prop-scrap').value = element.scrap_rate || '';
    document.getElementById('prop-available').value = element.available_time || '';
    document.getElementById('prop-shifts').value = element.shifts || '';

    selectedElement = element;
}


function showWorkCenterProperties(wc) {
    const panel = document.getElementById('properties-panel');
    panel.classList.add('visible');
    selectedElementType = 'workcenter';

    document.getElementById('prop-panel-title').textContent = 'Work Center';
    // Hide all non-relevant groups
    document.getElementById('prop-delay-group').style.display = 'none';
    document.getElementById('prop-dos-group').style.display = 'none';
    document.getElementById('prop-kaizen-text-group').style.display = 'none';
    document.getElementById('prop-kaizen-priority-group').style.display = 'none';
    document.getElementById('prop-kaizen-hypothesis-group').style.display = 'none';
    document.getElementById('prop-entity-detail-group').style.display = 'none';
    document.querySelectorAll('.prop-row').forEach(r => r.style.display = 'none');
    // Show WC fields
    document.getElementById('prop-wc-group').style.display = 'block';
    document.getElementById('prop-name').value = wc.name || 'Work Center';
    document.getElementById('prop-wc-width').value = wc.width || 280;
    document.getElementById('prop-wc-height').value = wc.height || 200;

    // Show effective CT and members
    const members = getWorkCenterMembers(wc.id);
    const effCT = getWorkCenterEffectiveCT(wc.id);
    document.getElementById('prop-wc-effective-ct').textContent =
        members.length > 0 ? `Effective C/T: ${formatTime(effCT)} (${members.length} machine${members.length !== 1 ? 's' : ''})` : 'Effective C/T: -';
    document.getElementById('prop-wc-members').textContent =
        members.length > 0 ? 'Members: ' + members.map(m => m.name).join(', ') : 'Members: drag process steps inside';

    selectedElement = wc;
}


function startDragElement(e, element, elementType = 'process') {
    e.stopPropagation();
    wasDragged = false;

    // Handle flow tool clicks
    if (currentTool === 'flow' && elementType === 'process') {
        handleFlowClick(element);
        return;
    }

    if (currentTool !== 'select') return;

    dragElement = element;
    selectedElementType = elementType;
    const rect = e.currentTarget.getBoundingClientRect();
    dragOffsetX = e.clientX - rect.left;
    dragOffsetY = e.clientY - rect.top;

    selectedElement = element;
    document.addEventListener('mousemove', dragElementMove);
    document.addEventListener('mouseup', dragElementEnd);
}


function startResizeWorkCenter(e, wc) {
    e.stopPropagation();
    e.preventDefault();
    resizingWC = wc;
    resizeStartX = e.clientX;
    resizeStartY = e.clientY;
    resizeStartW = wc.width || 280;
    resizeStartH = wc.height || 200;
    document.addEventListener('mousemove', resizeWorkCenterMove);
    document.addEventListener('mouseup', resizeWorkCenterEnd);
}

// =============================================================================
function updateCanvasTransform() {
    const elementsLayer = document.getElementById('elements-layer');
    const connectionsLayer = document.getElementById('connections-layer');
    const transform = `translate(${panX}, ${panY}) scale(${zoom})`;
    elementsLayer.setAttribute('transform', transform);
    connectionsLayer.setAttribute('transform', transform);
}


function zoomIn() {
    zoom = Math.min(4, zoom * 1.2);
    updateCanvasTransform();
}


function zoomOut() {
    zoom = Math.max(0.25, zoom * 0.8);
    updateCanvasTransform();
}

