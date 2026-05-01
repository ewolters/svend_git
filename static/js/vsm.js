// =============================================================================
// SVG Icon Paths (Svend style: stroke-based, 24x24 viewBox)
// =============================================================================
const VSM_ICONS = {
    customer: '<circle cx="9" cy="7" r="4"/><path d="M2 21v-2a4 4 0 0 1 4-4h6a4 4 0 0 1 4 4v2"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/>',
    supplier: '<path d="M4 21V11l5 3v-3l5 3V6h6v15"/><line x1="2" y1="21" x2="22" y2="21"/>',
    queue: '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    transport: '<rect x="1" y="3" width="15" height="13" rx="1"/><path d="M16 8h4l3 3v5h-7V8z"/><circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/>',
    batch: '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>'
};

function createSvgIcon(x, y, size, pathData, color) {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('x', x - size / 2);
    svg.setAttribute('y', y - size / 2);
    svg.setAttribute('width', size);
    svg.setAttribute('height', size);
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', color || 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.innerHTML = pathData;
    return svg;
}

// =============================================================================
// VSM State
// =============================================================================
let currentVSM = null;
let vsmId = null;
let currentProjectId = null;
let selectedElement = null;
let selectedElementType = null;  // 'process', 'inventory', 'kaizen', etc.
let currentTool = 'select';
let currentFlowType = null;  // 'push' or 'pull' when drawing flow
let flowSourceStep = null;   // First step when drawing connection
let zoom = 1;
let panX = 0;
let panY = 0;
let isPanning = false;
let lastMouseX = 0;
let lastMouseY = 0;

// Undo/Redo history
let vsmHistory = [];
let vsmHistoryIndex = -1;
const VSM_MAX_HISTORY = 50;

// Extract VSM ID from URL if present
const pathParts = window.location.pathname.split('/');
if (pathParts.length > 3 && pathParts[2] === 'vsm' && pathParts[3]) {
    vsmId = pathParts[3].replace('/', '');
}

// Check for project param in URL
const urlParams = new URLSearchParams(window.location.search);
const projectParam = urlParams.get('project');
if (projectParam) {
    currentProjectId = projectParam;
}

// =============================================================================
// Project Selector
// =============================================================================
async function setupProjectSelector() {
    const select = document.getElementById('project-select');
    const link = document.getElementById('project-link');

    try {
        const response = await fetch('/api/core/projects/', { credentials: 'include' });
        if (!response.ok) return;
        const projects = await response.json();

        // Populate dropdown
        projects.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = p.title;
            select.appendChild(opt);
        });

        // Set initial value from URL param or currentVSM.project
        if (currentProjectId) {
            select.value = currentProjectId;
            updateProjectLink(currentProjectId);
        }
    } catch (err) {
        console.error('Failed to load projects:', err);
    }

    // Handle selection change
    select.addEventListener('change', (e) => {
        currentProjectId = e.target.value || null;
        updateProjectLink(currentProjectId);
        // Update URL without reload
        const url = new URL(window.location);
        if (currentProjectId) {
            url.searchParams.set('project', currentProjectId);
        } else {
            url.searchParams.delete('project');
        }
        window.history.replaceState({}, '', url);
        // Save the project link
        if (currentVSM && vsmId) {
            saveVSM();
        }
    });
}

function updateProjectLink(projectId) {
    const link = document.getElementById('project-link');
    if (projectId) {
        link.href = `/app/investigations/?id=${projectId}`;
        link.style.display = 'flex';
    } else {
        link.style.display = 'none';
    }
}

// =============================================================================
// Initialization
// =============================================================================
document.addEventListener('DOMContentLoaded', async () => {
    await setupProjectSelector();

    if (vsmId) {
        await loadVSM(vsmId);
    } else {
        // Show empty state or list
        document.getElementById('empty-state').style.display = 'block';
        await loadVSMList();
    }

    setupEventListeners();
    setupDragAndDrop();
});

// =============================================================================
// API Functions
// =============================================================================
async function loadVSM(id) {
    try {
        const response = await fetch(`/api/vsm/${id}/`, { credentials: 'include' });
        if (!response.ok) throw new Error('VSM not found');
        const data = await response.json();
        currentVSM = data.vsm;

        // Set project from loaded VSM
        if (currentVSM.project_id) {
            currentProjectId = currentVSM.project_id;
            document.getElementById('project-select').value = currentProjectId;
            updateProjectLink(currentProjectId);
        }

        renderVSM();
        updateMetrics();
        updateProposalButton();
        document.getElementById('empty-state').style.display = 'none';
        // Seed undo history with initial state
        vsmHistory = [];
        vsmHistoryIndex = -1;
        saveVSMState();
        // Load hypothesis probabilities for kaizen bursts
        loadHypothesisProbabilities().then(() => renderVSM());
    } catch (err) {
        console.error('Load VSM error:', err);
    }
}

async function loadVSMList() {
    try {
        const response = await fetch('/api/vsm/', { credentials: 'include' });
        if (!response.ok) return;
        const data = await response.json();

        if (data.maps && data.maps.length > 0) {
            // Show list
            showVSMList(data.maps);
        }
    } catch (err) {
        console.error('Load VSM list error:', err);
    }
}

function saveVSMState() {
    if (!currentVSM) return;
    // Snapshot mutable arrays (deep copy)
    const snapshot = JSON.parse(JSON.stringify({
        process_steps: currentVSM.process_steps || [],
        inventory: currentVSM.inventory || [],
        kaizen_bursts: currentVSM.kaizen_bursts || [],
        material_flow: currentVSM.material_flow || [],
        information_flow: currentVSM.information_flow || [],
        customers: currentVSM.customers || [],
        suppliers: currentVSM.suppliers || [],
        work_centers: currentVSM.work_centers || [],
        customer_name: currentVSM.customer_name,
        customer_demand: currentVSM.customer_demand,
        supplier_name: currentVSM.supplier_name,
        supply_frequency: currentVSM.supply_frequency,
        takt_time: currentVSM.takt_time,
    }));
    // Truncate any redo states ahead of current index
    vsmHistory = vsmHistory.slice(0, vsmHistoryIndex + 1);
    vsmHistory.push(snapshot);
    if (vsmHistory.length > VSM_MAX_HISTORY) vsmHistory.shift();
    vsmHistoryIndex = vsmHistory.length - 1;
}

function undoVSM() {
    if (vsmHistoryIndex <= 0 || !currentVSM) return;
    vsmHistoryIndex--;
    applyVSMSnapshot(vsmHistory[vsmHistoryIndex]);
}

function redoVSM() {
    if (vsmHistoryIndex >= vsmHistory.length - 1 || !currentVSM) return;
    vsmHistoryIndex++;
    applyVSMSnapshot(vsmHistory[vsmHistoryIndex]);
}

function applyVSMSnapshot(snapshot) {
    const restored = JSON.parse(JSON.stringify(snapshot));
    Object.assign(currentVSM, restored);
    renderVSM();
    updateMetrics();
    saveVSM();
}

async function saveVSM() {
    if (!currentVSM || !vsmId) return;

    try {
        const payload = { ...currentVSM };
        if (currentProjectId) {
            payload.project_id = currentProjectId;
        }
        await fetch(`/api/vsm/${vsmId}/update/`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(payload)
        });
    } catch (err) {
        console.error('Save VSM error:', err);
    }
}

async function createNewVSM() {
    const name = document.getElementById('new-vsm-name').value || 'Untitled VSM';
    const productFamily = document.getElementById('new-vsm-product').value || '';
    const customerDemand = document.getElementById('new-vsm-demand').value || '';

    try {
        const payload = {
            name,
            product_family: productFamily,
            customer_demand: customerDemand
        };
        if (currentProjectId) {
            payload.project_id = currentProjectId;
        }
        const response = await fetch('/api/vsm/create/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('Failed to create VSM');
        const data = await response.json();
        svendTrack('feature_use', {category: 'vsm', action: 'create'});
        window.location.href = `/app/vsm/${data.id}/`;
    } catch (err) {
        console.error('Create VSM error:', err);
    }
}

// =============================================================================
// Rendering
// =============================================================================
function renderVSM() {
    if (!currentVSM) return;
    try {

    document.getElementById('vsm-name').textContent = currentVSM.name;

    const elementsLayer = document.getElementById('elements-layer');
    const connectionsLayer = document.getElementById('connections-layer');
    elementsLayer.innerHTML = '';
    connectionsLayer.innerHTML = '';

    // Render work centers FIRST (behind process steps)
    (currentVSM.work_centers || []).forEach(wc => {
        renderWorkCenter(wc, elementsLayer);
    });

    // Render process steps
    (currentVSM.process_steps || []).forEach(step => {
        renderProcessBox(step, elementsLayer);
    });

    // Render inventory
    (currentVSM.inventory || []).forEach(inv => {
        renderInventory(inv, elementsLayer);
    });

    // Render kaizen bursts
    (currentVSM.kaizen_bursts || []).forEach(burst => {
        renderKaizenBurst(burst, elementsLayer);
    });

    // Render customer/supplier
    renderCustomerSupplier(elementsLayer);

    // Render connections
    renderConnections(connectionsLayer);

    // Render lead time ladder
    renderLeadTimeLadder(elementsLayer);

    } catch (err) {
        console.error('renderVSM FAILED:', err);
    }
}

function renderProcessBox(step, layer) {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'vsm-element process-element');
    g.setAttribute('data-id', step.id);
    g.setAttribute('transform', `translate(${step.x}, ${step.y})`);

    // Box - taller to fit more metrics
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('width', '130');
    rect.setAttribute('height', '140');
    rect.setAttribute('fill', 'var(--bg-secondary)');
    rect.setAttribute('stroke', 'var(--accent-primary)');
    rect.setAttribute('stroke-width', '2');

    // Header
    const header = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    header.setAttribute('width', '130');
    header.setAttribute('height', '25');
    header.setAttribute('fill', 'var(--accent-primary)');

    const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    title.setAttribute('x', '65');
    title.setAttribute('y', '17');
    title.setAttribute('text-anchor', 'middle');
    title.setAttribute('fill', 'white');
    title.setAttribute('font-size', '11');
    title.setAttribute('font-weight', '500');
    title.textContent = step.name || 'Process';

    g.appendChild(rect);
    g.appendChild(header);
    g.appendChild(title);

    // Metrics - standard VSM data box format
    const metrics = [
        ['C/T', step.cycle_time ? formatTime(step.cycle_time) : '-'],
        ['C/O', step.changeover_time ? formatTime(step.changeover_time) : '-'],
        ['Uptime', step.uptime ? `${step.uptime}%` : '-'],
        ['Batch', step.batch_size || '-'],
        ['Scrap', step.scrap_rate ? `${step.scrap_rate}%` : '-'],
        ['Ops', step.operators || '-'],
        ['Shifts', step.shifts || '-'],
    ];

    metrics.forEach((m, i) => {
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', '8');
        label.setAttribute('y', 40 + i * 14);
        label.setAttribute('fill', 'var(--text-dim)');
        label.setAttribute('font-size', '9');
        label.textContent = m[0];

        const value = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        value.setAttribute('x', '122');
        value.setAttribute('y', 40 + i * 14);
        value.setAttribute('text-anchor', 'end');
        value.setAttribute('fill', 'var(--text-primary)');
        value.setAttribute('font-size', '9');
        value.setAttribute('font-weight', '500');
        value.textContent = m[1];

        g.appendChild(label);
        g.appendChild(value);
    });

    // Work center membership indicator
    if (step.work_center_id) {
        const wcDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        wcDot.setAttribute('cx', '120');
        wcDot.setAttribute('cy', '132');
        wcDot.setAttribute('r', '4');
        wcDot.setAttribute('fill', 'var(--accent-primary)');
        wcDot.setAttribute('opacity', '0.5');
        g.appendChild(wcDot);
    }

    // --- Bottleneck & takt flags ---
    const flags = step.flags || {};
    if (flags.is_bottleneck) {
        // Red stroke on box
        rect.setAttribute('stroke', '#e74c3c');
        rect.setAttribute('stroke-width', '2.5');
        // Red "B" badge top-right
        const badge = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        badge.setAttribute('cx', '125'); badge.setAttribute('cy', '5');
        badge.setAttribute('r', '7'); badge.setAttribute('fill', '#e74c3c');
        badge.setAttribute('stroke', 'var(--bg-primary)'); badge.setAttribute('stroke-width', '1.5');
        g.appendChild(badge);
        const bText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        bText.setAttribute('x', '125'); bText.setAttribute('y', '8.5');
        bText.setAttribute('text-anchor', 'middle'); bText.setAttribute('fill', 'white');
        bText.setAttribute('font-size', '8'); bText.setAttribute('font-weight', '700');
        bText.textContent = 'B';
        g.appendChild(bText);
    } else if (flags.exceeds_takt) {
        // Orange stroke for exceeds-takt (not bottleneck)
        rect.setAttribute('stroke', '#f39c12');
        rect.setAttribute('stroke-width', '2.5');
    }

    // --- Annotation badges (colored dots at bottom) ---
    const annotations = step.annotations || [];
    if (annotations.length > 0) {
        const statusColors = { green: '#4a9f6e', yellow: '#f1c40f', red: '#e74c3c' };
        annotations.forEach((a, i) => {
            if (i >= 8) return; // max 8 dots
            const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            dot.setAttribute('cx', String(8 + i * 12));
            dot.setAttribute('cy', '136');
            dot.setAttribute('r', '3.5');
            dot.setAttribute('fill', statusColors[a.status] || '#666');
            dot.setAttribute('stroke', 'var(--bg-secondary)'); dot.setAttribute('stroke-width', '1');
            g.appendChild(dot);
        });
    }

    // Make draggable; single click = metrics, double click = properties
    g.addEventListener('mousedown', (e) => startDragElement(e, step, 'process'));
    g.addEventListener('click', (e) => {
        if (e.detail === 1 && !wasDragged) {
            clearTimeout(stepMetricsClickTimer);
            stepMetricsClickTimer = setTimeout(() => showStepMetrics(step), 200);
        }
    });
    // Store step ref on the DOM element for canvas-level dblclick dispatch
    g._vsmStep = step;

    layer.appendChild(g);
}

function formatTime(seconds) {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${(seconds/60).toFixed(1)}m`;
    return `${(seconds/3600).toFixed(1)}h`;
}

function renderWorkCenter(wc, layer) {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'vsm-element workcenter-element');
    g.setAttribute('data-id', wc.id);
    g.setAttribute('transform', `translate(${wc.x}, ${wc.y})`);

    // Dotted-line rectangle
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('width', wc.width || 280);
    rect.setAttribute('height', wc.height || 200);
    rect.setAttribute('fill', 'none');
    rect.setAttribute('stroke', 'var(--accent-primary)');
    rect.setAttribute('stroke-width', '2');
    rect.setAttribute('stroke-dasharray', '8 4');
    rect.setAttribute('rx', '6');
    rect.setAttribute('opacity', '0.6');
    g.appendChild(rect);

    // Subtle fill to show containment
    const fill = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    fill.setAttribute('width', wc.width || 280);
    fill.setAttribute('height', wc.height || 200);
    fill.setAttribute('fill', 'var(--accent-primary)');
    fill.setAttribute('opacity', '0.04');
    fill.setAttribute('rx', '6');
    g.appendChild(fill);

    // Name label (top-left, inside the box)
    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', '8');
    label.setAttribute('y', '-6');
    label.setAttribute('fill', 'var(--accent-primary)');
    label.setAttribute('font-size', '11');
    label.setAttribute('font-weight', '600');
    label.setAttribute('opacity', '0.8');
    label.textContent = wc.name || 'Work Center';
    g.appendChild(label);

    // Effective CT badge (top-right)
    const effCT = getWorkCenterEffectiveCT(wc.id);
    if (effCT > 0) {
        const ctBadge = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        ctBadge.setAttribute('x', (wc.width || 280) - 8);
        ctBadge.setAttribute('y', '-6');
        ctBadge.setAttribute('text-anchor', 'end');
        ctBadge.setAttribute('fill', 'var(--text-dim)');
        ctBadge.setAttribute('font-size', '10');
        ctBadge.textContent = `Eff. C/T: ${formatTime(effCT)}`;
        g.appendChild(ctBadge);
    }

    // Resize handle (bottom-right corner)
    const handle = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    handle.setAttribute('x', (wc.width || 280) - 12);
    handle.setAttribute('y', (wc.height || 200) - 12);
    handle.setAttribute('width', '12');
    handle.setAttribute('height', '12');
    handle.setAttribute('fill', 'var(--accent-primary)');
    handle.setAttribute('opacity', '0.3');
    handle.setAttribute('rx', '2');
    handle.setAttribute('cursor', 'nwse-resize');
    handle.addEventListener('mousedown', (e) => startResizeWorkCenter(e, wc));
    g.appendChild(handle);

    // Drag on main rect, double-click for properties
    g.addEventListener('mousedown', (e) => {
        if (e.target === handle) return; // handled by resize
        startDragElement(e, wc, 'workcenter');
    });
    g._vsmWC = wc;

    layer.appendChild(g);
}

function getWorkCenterEffectiveCT(wcId) {
    if (!currentVSM) return 0;
    const members = (currentVSM.process_steps || []).filter(s => s.work_center_id === wcId);
    if (members.length === 0) return 0;
    const rateSum = members.reduce((sum, s) => {
        const ct = s.cycle_time || 0;
        return ct > 0 ? sum + (1.0 / ct) : sum;
    }, 0);
    return rateSum > 0 ? 1.0 / rateSum : 0;
}

function getWorkCenterMembers(wcId) {
    if (!currentVSM) return [];
    return (currentVSM.process_steps || []).filter(s => s.work_center_id === wcId);
}

function associateStepsToWorkCenters() {
    if (!currentVSM) return;
    const wcs = currentVSM.work_centers || [];
    const steps = currentVSM.process_steps || [];
    steps.forEach(step => {
        const stepCX = step.x + 65; // center of 130px process box
        const stepCY = step.y + 70; // center of 140px process box
        let found = false;
        for (const wc of wcs) {
            if (stepCX >= wc.x && stepCX <= wc.x + (wc.width || 280) &&
                stepCY >= wc.y && stepCY <= wc.y + (wc.height || 200)) {
                step.work_center_id = wc.id;
                found = true;
                break;
            }
        }
        if (!found && step.work_center_id) {
            delete step.work_center_id;
        }
    });
}

let resizingWC = null;
let resizeStartX = 0;
let resizeStartY = 0;
let resizeStartW = 0;
let resizeStartH = 0;

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

function resizeWorkCenterMove(e) {
    if (!resizingWC) return;
    const dx = (e.clientX - resizeStartX) / zoom;
    const dy = (e.clientY - resizeStartY) / zoom;
    resizingWC.width = Math.max(160, resizeStartW + dx);
    resizingWC.height = Math.max(100, resizeStartH + dy);
    renderVSM();
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

function renderInventory(inv, layer) {
    // Route to specialized renderers for special types
    if (inv.is_supermarket || inv.delay_type === 'supermarket') {
        renderSupermarket(inv, layer);
        return;
    }
    if (inv.is_fifo || inv.delay_type === 'fifo') {
        renderFIFO(inv, layer);
        return;
    }

    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'vsm-element inventory-element');
    g.setAttribute('data-id', inv.id);
    g.setAttribute('transform', `translate(${inv.x}, ${inv.y})`);

    // Color based on delay type
    const delayColors = {
        'inventory': 'var(--warning)',
        'queue': '#f59e0b',
        'transport': '#8b5cf6',
        'batch': '#ec4899'
    };
    const color = delayColors[inv.delay_type] || delayColors['inventory'];

    // Triangle
    const triangle = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    triangle.setAttribute('points', '30,0 60,52 0,52');
    triangle.setAttribute('fill', color);
    triangle.setAttribute('opacity', '0.8');
    g.appendChild(triangle);

    // Icon overlay for special delay types
    if (inv.delay_type && inv.delay_type !== 'inventory') {
        const delayIcons = {
            'queue': VSM_ICONS.queue,
            'transport': VSM_ICONS.transport,
            'batch': VSM_ICONS.batch
        };
        const pathData = delayIcons[inv.delay_type];
        if (pathData) {
            g.appendChild(createSvgIcon(30, 22, 18, pathData, 'white'));
        }
    }

    // Days of supply text
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', '30');
    text.setAttribute('y', inv.delay_type && inv.delay_type !== 'inventory' ? '46' : '40');
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('fill', 'var(--bg-primary)');
    text.setAttribute('font-size', '10');
    text.setAttribute('font-weight', '600');
    const displayDays = inv.days_of_supply || inv.computed_days;
    text.textContent = displayDays ? `${displayDays}d` : inv.quantity || 'I';
    if (!inv.days_of_supply && inv.computed_days) {
        text.setAttribute('font-style', 'italic');  // italic = auto-computed
    }
    g.appendChild(text);

    // Type label below
    if (inv.delay_type && inv.delay_type !== 'inventory') {
        const typeLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        typeLabel.setAttribute('x', '30');
        typeLabel.setAttribute('y', '65');
        typeLabel.setAttribute('text-anchor', 'middle');
        typeLabel.setAttribute('fill', color);
        typeLabel.setAttribute('font-size', '8');
        const labels = {
            'queue': 'Queue',
            'transport': 'Transport',
            'batch': 'Batch'
        };
        typeLabel.textContent = labels[inv.delay_type] || '';
        g.appendChild(typeLabel);
    }

    g.addEventListener('mousedown', (e) => startDragElement(e, inv, 'inventory'));
    g._vsmInv = inv;

    layer.appendChild(g);
}

function renderSupermarket(inv, layer) {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'vsm-element supermarket-element');
    g.setAttribute('data-id', inv.id);
    g.setAttribute('transform', `translate(${inv.x}, ${inv.y})`);

    // Supermarket icon (shelves pattern)
    const box = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    box.setAttribute('width', '50');
    box.setAttribute('height', '60');
    box.setAttribute('fill', 'var(--bg-tertiary)');
    box.setAttribute('stroke', '#06b6d4');
    box.setAttribute('stroke-width', '2');
    g.appendChild(box);

    // Shelf lines
    for (let i = 1; i <= 3; i++) {
        const shelf = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        shelf.setAttribute('x1', '0');
        shelf.setAttribute('y1', i * 15);
        shelf.setAttribute('x2', '50');
        shelf.setAttribute('y2', i * 15);
        shelf.setAttribute('stroke', '#06b6d4');
        shelf.setAttribute('stroke-width', '1');
        g.appendChild(shelf);
    }

    // Label
    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', '25');
    label.setAttribute('y', '75');
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('fill', '#06b6d4');
    label.setAttribute('font-size', '9');
    label.textContent = 'Supermarket';
    g.appendChild(label);

    g.addEventListener('mousedown', (e) => startDragElement(e, inv, 'inventory'));
    g._vsmInv = inv;

    layer.appendChild(g);
}

function renderFIFO(inv, layer) {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'vsm-element fifo-element');
    g.setAttribute('data-id', inv.id);
    g.setAttribute('transform', `translate(${inv.x}, ${inv.y})`);

    // FIFO lane (horizontal rectangle with arrow)
    const lane = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    lane.setAttribute('width', '80');
    lane.setAttribute('height', '30');
    lane.setAttribute('fill', 'var(--bg-tertiary)');
    lane.setAttribute('stroke', '#10b981');
    lane.setAttribute('stroke-width', '2');
    g.appendChild(lane);

    // FIFO text
    const fifoText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    fifoText.setAttribute('x', '40');
    fifoText.setAttribute('y', '20');
    fifoText.setAttribute('text-anchor', 'middle');
    fifoText.setAttribute('fill', '#10b981');
    fifoText.setAttribute('font-size', '10');
    fifoText.setAttribute('font-weight', '600');
    fifoText.textContent = 'FIFO';
    g.appendChild(fifoText);

    // Arrow indicating direction
    const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    arrow.setAttribute('d', 'M65,15 L75,15 L70,8 M75,15 L70,22');
    arrow.setAttribute('stroke', '#10b981');
    arrow.setAttribute('stroke-width', '2');
    arrow.setAttribute('fill', 'none');
    g.appendChild(arrow);

    // Max quantity label
    if (inv.max_quantity) {
        const maxLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        maxLabel.setAttribute('x', '40');
        maxLabel.setAttribute('y', '45');
        maxLabel.setAttribute('text-anchor', 'middle');
        maxLabel.setAttribute('fill', 'var(--text-dim)');
        maxLabel.setAttribute('font-size', '8');
        maxLabel.textContent = `Max: ${inv.max_quantity}`;
        g.appendChild(maxLabel);
    }

    g.addEventListener('mousedown', (e) => startDragElement(e, inv, 'inventory'));
    g._vsmInv = inv;

    layer.appendChild(g);
}

function renderKaizenBurst(burst, layer) {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'vsm-element kaizen-element');
    g.setAttribute('data-id', burst.id);
    g.setAttribute('transform', `translate(${burst.x}, ${burst.y})`);

    // Star burst
    const star = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    const points = [];
    const outerR = 40;
    const innerR = 20;
    for (let i = 0; i < 10; i++) {
        const r = i % 2 === 0 ? outerR : innerR;
        const angle = (i * Math.PI / 5) - Math.PI / 2;
        points.push(`${40 + r * Math.cos(angle)},${40 + r * Math.sin(angle)}`);
    }
    star.setAttribute('points', points.join(' '));
    star.setAttribute('fill', burst.priority === 'high' ? 'var(--error)' : 'var(--warning)');

    // Text — dark fill for readability on orange/red background
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', '40');
    text.setAttribute('y', '38');
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('dominant-baseline', 'middle');
    text.setAttribute('fill', '#1a1a2e');
    text.setAttribute('font-size', '8');
    text.setAttribute('font-weight', '600');

    // Word-wrap into two lines if needed
    const label = burst.text || '';
    if (label.length > 12) {
        const mid = label.lastIndexOf(' ', 12);
        const split = mid > 0 ? mid : 12;
        const tspan1 = document.createElementNS('http://www.w3.org/2000/svg', 'tspan');
        tspan1.setAttribute('x', '40');
        tspan1.setAttribute('dy', '-5');
        tspan1.textContent = label.slice(0, split);
        const tspan2 = document.createElementNS('http://www.w3.org/2000/svg', 'tspan');
        tspan2.setAttribute('x', '40');
        tspan2.setAttribute('dy', '11');
        tspan2.textContent = label.slice(split).trim().slice(0, 14);
        text.appendChild(tspan1);
        text.appendChild(tspan2);
    } else {
        text.textContent = label;
    }

    g.appendChild(star);
    g.appendChild(text);

    // Probability badge if linked to hypothesis
    if (burst.hypothesis_id && burst._hypothesis_prob != null) {
        const prob = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        prob.setAttribute('x', '40');
        prob.setAttribute('y', '70');
        prob.setAttribute('text-anchor', 'middle');
        prob.setAttribute('fill', 'var(--accent)');
        prob.setAttribute('font-size', '9');
        prob.setAttribute('font-weight', '600');
        prob.textContent = `P=${Math.round(burst._hypothesis_prob * 100)}%`;
        g.appendChild(prob);
    }

    // Realized savings badge (written back from Hoshin)
    if (burst.realized_savings) {
        const yPos = (burst.hypothesis_id && burst._hypothesis_prob != null) ? 82 : 70;
        const savBadge = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        savBadge.setAttribute('x', '40');
        savBadge.setAttribute('y', String(yPos));
        savBadge.setAttribute('text-anchor', 'middle');
        savBadge.setAttribute('fill', '#6fcf97');
        savBadge.setAttribute('font-size', '8');
        savBadge.setAttribute('font-weight', '600');
        const amt = Math.abs(burst.realized_savings);
        savBadge.textContent = amt >= 1000 ? `$${(amt/1000).toFixed(1)}K` : `$${amt.toFixed(0)}`;
        g.appendChild(savBadge);
    }

    g.addEventListener('mousedown', (e) => startDragElement(e, burst, 'kaizen'));
    g._vsmBurst = burst;

    layer.appendChild(g);
}

function renderCustomerSupplier(layer) {
    if (!currentVSM) return;

    // Migrate legacy single-field data into arrays on first load
    if ((!currentVSM.customers || currentVSM.customers.length === 0) && currentVSM.customer_name) {
        currentVSM.customers = [{
            id: 'legacy-customer',
            name: currentVSM.customer_name || 'Customer',
            detail: currentVSM.customer_demand || '',
            x: 850, y: 50
        }];
    }
    if ((!currentVSM.suppliers || currentVSM.suppliers.length === 0) && currentVSM.supplier_name) {
        currentVSM.suppliers = [{
            id: 'legacy-supplier',
            name: currentVSM.supplier_name || 'Supplier',
            detail: currentVSM.supply_frequency || '',
            x: 50, y: 50
        }];
    }

    // Render all customers
    (currentVSM.customers || []).forEach(ent => {
        renderEntityBox(layer, ent, 'customer', VSM_ICONS.customer);
    });

    // Render all suppliers
    (currentVSM.suppliers || []).forEach(ent => {
        renderEntityBox(layer, ent, 'supplier', VSM_ICONS.supplier);
    });
}

function renderEntityBox(layer, ent, entityType, icon) {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'vsm-element entity-element');
    g.setAttribute('data-id', ent.id);
    g.setAttribute('transform', `translate(${ent.x}, ${ent.y})`);

    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('width', '100');
    rect.setAttribute('height', '80');
    rect.setAttribute('fill', 'var(--bg-tertiary)');
    rect.setAttribute('stroke', 'var(--accent-blue)');
    rect.setAttribute('stroke-width', '2');
    rect.setAttribute('rx', '4');
    rect.style.cursor = 'move';

    const iconEl = createSvgIcon(50, 20, 24, icon, 'var(--accent-blue)');

    const nameEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    nameEl.setAttribute('x', '50');
    nameEl.setAttribute('y', '52');
    nameEl.setAttribute('text-anchor', 'middle');
    nameEl.setAttribute('fill', 'var(--text-primary)');
    nameEl.setAttribute('font-size', '11');
    nameEl.setAttribute('font-weight', '600');
    nameEl.textContent = ent.name || entityType;

    const detailEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    detailEl.setAttribute('x', '50');
    detailEl.setAttribute('y', '68');
    detailEl.setAttribute('text-anchor', 'middle');
    detailEl.setAttribute('fill', 'var(--text-dim)');
    detailEl.setAttribute('font-size', '9');
    detailEl.textContent = ent.detail || '';

    g.appendChild(rect);
    g.appendChild(iconEl);
    g.appendChild(nameEl);
    g.appendChild(detailEl);

    // Drag to reposition
    g.addEventListener('mousedown', (e) => {
        if (currentTool !== 'select') return;
        e.stopPropagation();
        const startMouse = { x: e.clientX, y: e.clientY };
        const startX = ent.x, startY = ent.y;

        function onMove(ev) {
            ent.x = startX + (ev.clientX - startMouse.x) / zoom;
            ent.y = startY + (ev.clientY - startMouse.y) / zoom;
            renderVSM();
        }
        function onUp() {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            saveVSM();
        }
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });

    // Store ref for canvas-level dblclick dispatch
    g._vsmEntity = ent;
    g._vsmEntityType = entityType;

    layer.appendChild(g);
}

function renderConnections(layer) {
    if (!currentVSM) return;

    // Add arrow markers to defs
    const svg = document.getElementById('vsm-canvas');
    let defs = svg.querySelector('defs');
    if (!defs.querySelector('#arrow-push')) {
        // Push arrow (striped, gray)
        const pushMarker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        pushMarker.setAttribute('id', 'arrow-push');
        pushMarker.setAttribute('markerWidth', '10');
        pushMarker.setAttribute('markerHeight', '10');
        pushMarker.setAttribute('refX', '9');
        pushMarker.setAttribute('refY', '3');
        pushMarker.setAttribute('orient', 'auto');
        pushMarker.setAttribute('markerUnits', 'strokeWidth');
        const pushPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        pushPath.setAttribute('d', 'M0,0 L0,6 L9,3 z');
        pushPath.setAttribute('fill', 'var(--text-secondary)');
        pushMarker.appendChild(pushPath);
        defs.appendChild(pushMarker);

        // Pull arrow (solid, green)
        const pullMarker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        pullMarker.setAttribute('id', 'arrow-pull');
        pullMarker.setAttribute('markerWidth', '10');
        pullMarker.setAttribute('markerHeight', '10');
        pullMarker.setAttribute('refX', '9');
        pullMarker.setAttribute('refY', '3');
        pullMarker.setAttribute('orient', 'auto');
        pullMarker.setAttribute('markerUnits', 'strokeWidth');
        const pullPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        pullPath.setAttribute('d', 'M0,0 L0,6 L9,3 z');
        pullPath.setAttribute('fill', 'var(--success)');
        pullMarker.appendChild(pullPath);
        defs.appendChild(pullMarker);

        // Kanban signal marker (for pull)
        const kanbanMarker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        kanbanMarker.setAttribute('id', 'kanban-signal');
        kanbanMarker.setAttribute('markerWidth', '12');
        kanbanMarker.setAttribute('markerHeight', '12');
        kanbanMarker.setAttribute('refX', '6');
        kanbanMarker.setAttribute('refY', '6');
        kanbanMarker.setAttribute('orient', 'auto');
        const kanbanRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        kanbanRect.setAttribute('x', '2');
        kanbanRect.setAttribute('y', '2');
        kanbanRect.setAttribute('width', '8');
        kanbanRect.setAttribute('height', '8');
        kanbanRect.setAttribute('fill', 'var(--success)');
        kanbanRect.setAttribute('stroke', 'white');
        kanbanRect.setAttribute('stroke-width', '1');
        kanbanMarker.appendChild(kanbanRect);
        defs.appendChild(kanbanMarker);
    }

    // Material flow
    (currentVSM.material_flow || []).forEach(conn => {
        const fromStep = currentVSM.process_steps.find(s => s.id === conn.from_step_id);
        const toStep = currentVSM.process_steps.find(s => s.id === conn.to_step_id);
        if (!fromStep || !toStep) return;

        const isPull = conn.type === 'pull';
        const x1 = fromStep.x + 130;
        const y1 = fromStep.y + 70;
        const x2 = toStep.x;
        const y2 = toStep.y + 70;

        if (isPull) {
            // Pull flow: solid green arrow with kanban signal above
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', x1);
            line.setAttribute('y1', y1);
            line.setAttribute('x2', x2);
            line.setAttribute('y2', y2);
            line.setAttribute('stroke', 'var(--success)');
            line.setAttribute('stroke-width', '3');
            line.setAttribute('marker-end', 'url(#arrow-pull)');
            layer.appendChild(line);

            // Kanban signal (small square above the line midpoint)
            const midX = (x1 + x2) / 2;
            const midY = y1 - 15;
            const signal = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            signal.setAttribute('x', midX - 6);
            signal.setAttribute('y', midY - 6);
            signal.setAttribute('width', '12');
            signal.setAttribute('height', '12');
            signal.setAttribute('fill', 'var(--success)');
            signal.setAttribute('stroke', 'white');
            signal.setAttribute('stroke-width', '1');
            layer.appendChild(signal);

            // "K" label on signal
            const kLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            kLabel.setAttribute('x', midX);
            kLabel.setAttribute('y', midY + 3);
            kLabel.setAttribute('text-anchor', 'middle');
            kLabel.setAttribute('fill', 'white');
            kLabel.setAttribute('font-size', '8');
            kLabel.setAttribute('font-weight', '600');
            kLabel.textContent = 'K';
            layer.appendChild(kLabel);
        } else {
            // Push flow: striped/dashed gray arrow
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', x1);
            line.setAttribute('y1', y1);
            line.setAttribute('x2', x2);
            line.setAttribute('y2', y2);
            line.setAttribute('stroke', 'var(--text-secondary)');
            line.setAttribute('stroke-width', '4');
            line.setAttribute('stroke-dasharray', '10,5');
            line.setAttribute('marker-end', 'url(#arrow-push)');
            layer.appendChild(line);
        }
    });

    // Information flow (dashed blue lines, typically upward)
    (currentVSM.information_flow || []).forEach(conn => {
        const fromStep = currentVSM.process_steps.find(s => s.id === conn.from_step_id);
        const toStep = currentVSM.process_steps.find(s => s.id === conn.to_step_id);
        if (!fromStep || !toStep) return;

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', fromStep.x + 65);
        line.setAttribute('y1', fromStep.y);
        line.setAttribute('x2', toStep.x + 65);
        line.setAttribute('y2', toStep.y);
        line.setAttribute('stroke', 'var(--accent-blue)');
        line.setAttribute('stroke-width', '2');
        line.setAttribute('stroke-dasharray', '5,5');
        layer.appendChild(line);
    });
}

function renderLeadTimeLadder(layer) {
    if (!currentVSM) return;

    const steps = currentVSM.process_steps || [];
    const inventory = currentVSM.inventory || [];
    if (steps.length === 0) return;

    // Sort steps and inventory by x position
    const sortedSteps = [...steps].sort((a, b) => a.x - b.x);
    const sortedInventory = [...inventory].sort((a, b) => a.x - b.x);

    // Find baseline Y (below all process boxes)
    const wcs = currentVSM.work_centers || [];
    const wcBottoms = wcs.length > 0 ? wcs.map(w => w.y + (w.height || 200)) : [0];
    const maxY = Math.max(...steps.map(s => s.y + 140), ...inventory.map(i => i.y + 60), ...wcBottoms) + 60;
    const baselineY = maxY + 30;
    const segmentHeight = 25;
    const minX = Math.min(...steps.map(s => s.x), ...inventory.map(i => i.x)) - 20;
    const maxX = Math.max(...steps.map(s => s.x + 130)) + 20;

    // Create ladder group
    const ladderG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    ladderG.setAttribute('class', 'lead-time-ladder');

    // Draw baseline
    const baseline = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    baseline.setAttribute('x1', minX);
    baseline.setAttribute('y1', baselineY);
    baseline.setAttribute('x2', maxX + 100);
    baseline.setAttribute('y2', baselineY);
    baseline.setAttribute('stroke', 'var(--text-dim)');
    baseline.setAttribute('stroke-width', '2');
    ladderG.appendChild(baseline);

    let totalCT = 0;
    let totalWait = 0;

    // Draw wait time segments for each inventory item (aligned to inventory position)
    sortedInventory.forEach(inv => {
        const waitDays = inv.days_of_supply || inv.computed_days || 0;
        if (waitDays > 0) {
            totalWait += waitDays;
            const invCenterX = inv.x + 30; // Center of inventory triangle
            const segWidth = 50;

            // Color based on delay type
            const delayColors = {
                'inventory': 'var(--warning)',
                'queue': '#f59e0b',
                'transport': '#8b5cf6',
                'batch': '#ec4899',
                'supermarket': '#06b6d4'
            };
            const color = delayColors[inv.delay_type] || delayColors['inventory'];

            // Elevated rectangle (above baseline) - centered on inventory
            const waitRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            waitRect.setAttribute('x', invCenterX - segWidth/2);
            waitRect.setAttribute('y', baselineY - segmentHeight);
            waitRect.setAttribute('width', segWidth);
            waitRect.setAttribute('height', segmentHeight);
            waitRect.setAttribute('fill', color);
            waitRect.setAttribute('opacity', '0.7');
            ladderG.appendChild(waitRect);

            // Wait time label
            const waitLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            waitLabel.setAttribute('x', invCenterX);
            waitLabel.setAttribute('y', baselineY - segmentHeight - 5);
            waitLabel.setAttribute('text-anchor', 'middle');
            waitLabel.setAttribute('fill', color);
            waitLabel.setAttribute('font-size', '10');
            waitLabel.setAttribute('font-weight', '500');
            waitLabel.textContent = `${waitDays}d`;
            ladderG.appendChild(waitLabel);
        }
    });

    // Draw cycle time segments — work center members get one combined segment
    const renderedWCs = new Set();
    sortedSteps.forEach(step => {
        const ct = step.cycle_time || 0;
        if (ct <= 0) return;

        if (step.work_center_id) {
            // Render work center as one combined segment (only once)
            if (renderedWCs.has(step.work_center_id)) return;
            renderedWCs.add(step.work_center_id);

            const wc = (currentVSM.work_centers || []).find(w => w.id === step.work_center_id);
            const effCT = getWorkCenterEffectiveCT(step.work_center_id);
            if (effCT <= 0) return;
            totalCT += effCT;

            const wcX = wc ? wc.x : step.x;
            const segWidth = wc ? (wc.width || 280) : 130;

            const ctRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            ctRect.setAttribute('x', wcX);
            ctRect.setAttribute('y', baselineY);
            ctRect.setAttribute('width', segWidth);
            ctRect.setAttribute('height', segmentHeight);
            ctRect.setAttribute('fill', 'var(--accent-primary)');
            ctRect.setAttribute('opacity', '0.7');
            ctRect.setAttribute('stroke', 'var(--accent-primary)');
            ctRect.setAttribute('stroke-width', '1');
            ctRect.setAttribute('stroke-dasharray', '4 2');
            ladderG.appendChild(ctRect);

            const ctLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            ctLabel.setAttribute('x', wcX + segWidth/2);
            ctLabel.setAttribute('y', baselineY + segmentHeight + 12);
            ctLabel.setAttribute('text-anchor', 'middle');
            ctLabel.setAttribute('fill', 'var(--accent-primary)');
            ctLabel.setAttribute('font-size', '10');
            ctLabel.setAttribute('font-weight', '500');
            ctLabel.textContent = `${formatTime(effCT)} (eff.)`;
            ladderG.appendChild(ctLabel);
        } else {
            totalCT += ct;
            const segWidth = 130;

            // Depressed rectangle (below baseline)
            const ctRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            ctRect.setAttribute('x', step.x);
            ctRect.setAttribute('y', baselineY);
            ctRect.setAttribute('width', segWidth);
            ctRect.setAttribute('height', segmentHeight);
            ctRect.setAttribute('fill', 'var(--accent-primary)');
            ctRect.setAttribute('opacity', '0.7');
            ladderG.appendChild(ctRect);

            // Cycle time label
            const ctLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            ctLabel.setAttribute('x', step.x + segWidth/2);
            ctLabel.setAttribute('y', baselineY + segmentHeight + 12);
            ctLabel.setAttribute('text-anchor', 'middle');
            ctLabel.setAttribute('fill', 'var(--accent-primary)');
            ctLabel.setAttribute('font-size', '10');
            ctLabel.setAttribute('font-weight', '500');
            ctLabel.textContent = formatTime(ct);
            ladderG.appendChild(ctLabel);
        }
    });

    // Draw totals box
    const totalsX = maxX + 40;
    const totalsG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    totalsG.setAttribute('transform', `translate(${totalsX}, ${baselineY - 40})`);

    // Lead time (top - wait)
    const ltLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    ltLabel.setAttribute('x', '0');
    ltLabel.setAttribute('y', '0');
    ltLabel.setAttribute('fill', 'var(--warning)');
    ltLabel.setAttribute('font-size', '10');
    ltLabel.textContent = 'Lead Time:';
    totalsG.appendChild(ltLabel);

    const ltValue = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    ltValue.setAttribute('x', '0');
    ltValue.setAttribute('y', '14');
    ltValue.setAttribute('fill', 'var(--warning)');
    ltValue.setAttribute('font-size', '14');
    ltValue.setAttribute('font-weight', '600');
    const totalLT = totalWait + (totalCT / 86400);
    ltValue.textContent = totalLT.toFixed(1) + ' days';
    totalsG.appendChild(ltValue);

    // Process time (bottom - value add)
    const ptLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    ptLabel.setAttribute('x', '0');
    ptLabel.setAttribute('y', '40');
    ptLabel.setAttribute('fill', 'var(--accent-primary)');
    ptLabel.setAttribute('font-size', '10');
    ptLabel.textContent = 'Process Time:';
    totalsG.appendChild(ptLabel);

    const ptValue = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    ptValue.setAttribute('x', '0');
    ptValue.setAttribute('y', '54');
    ptValue.setAttribute('fill', 'var(--accent-primary)');
    ptValue.setAttribute('font-size', '14');
    ptValue.setAttribute('font-weight', '600');
    ptValue.textContent = formatTime(totalCT);
    totalsG.appendChild(ptValue);

    // PCE
    const pce = totalLT > 0 ? ((totalCT / 86400) / totalLT * 100) : 0;
    const pceLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    pceLabel.setAttribute('x', '0');
    pceLabel.setAttribute('y', '76');
    pceLabel.setAttribute('fill', 'var(--text-dim)');
    pceLabel.setAttribute('font-size', '10');
    pceLabel.textContent = `PCE: ${pce.toFixed(1)}%`;
    totalsG.appendChild(pceLabel);

    ladderG.appendChild(totalsG);
    layer.appendChild(ladderG);
}

// =============================================================================
// Interaction
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

    // Canvas-level dblclick — bypasses stopPropagation on mousedown
    canvas.addEventListener('dblclick', (e) => {
        let el = e.target;
        while (el && el !== canvas) {
            if (el._vsmStep) {
                clearTimeout(stepMetricsClickTimer);
                closeStepMetrics();
                showProperties(el._vsmStep);
                return;
            }
            if (el._vsmWC) { showWorkCenterProperties(el._vsmWC); return; }
            if (el._vsmInv) { showInventoryProperties(el._vsmInv); return; }
            if (el._vsmBurst) { showKaizenProperties(el._vsmBurst); return; }
            if (el._vsmEntity) { showEntityProperties(el._vsmEntity, el._vsmEntityType); return; }
            el = el.parentNode;
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

function startDragElement(e, element, elementType = 'process') {
    wasDragged = false;

    // Handle flow tool clicks
    if (currentTool === 'flow' && elementType === 'process') {
        e.stopPropagation();
        handleFlowClick(element);
        return;
    }

    if (currentTool !== 'select') return;

    e.stopPropagation();
    dragElement = element;
    selectedElementType = elementType;
    const rect = e.currentTarget.getBoundingClientRect();
    dragOffsetX = e.clientX - rect.left;
    dragOffsetY = e.clientY - rect.top;

    selectedElement = element;
    document.addEventListener('mousemove', dragElementMove);
    document.addEventListener('mouseup', dragElementEnd);
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

    // Detect appropriate time unit from stored seconds value
    const ctSec = element.cycle_time || 0;
    let timeUnit = 'sec';
    if (ctSec >= 3600) timeUnit = 'hr';
    else if (ctSec >= 120) timeUnit = 'min';
    const tuEl = document.getElementById('prop-time-unit');
    if (tuEl) {
        tuEl.value = timeUnit;
        updateTimeUnitLabels(timeUnit);
    }

    // Convert stored seconds to display unit
    const divisor = timeUnit === 'hr' ? 3600 : timeUnit === 'min' ? 60 : 1;
    document.getElementById('prop-cycle-time').value = ctSec ? (ctSec / divisor) : '';
    document.getElementById('prop-changeover').value = element.changeover_time ? (element.changeover_time / divisor) : '';

    document.getElementById('prop-uptime').value = element.uptime || '';
    document.getElementById('prop-operators').value = element.operators || '';
    document.getElementById('prop-demand-rate').value = element.demand_rate || '';
    document.getElementById('prop-demand-unit').value = element.demand_unit || '';
    document.getElementById('prop-batch').value = element.batch_size || '';
    document.getElementById('prop-batch-process').checked = !!element.batch_process;
    document.getElementById('prop-unit-cost').value = element.unit_cost || '';
    document.getElementById('prop-setup-cost').value = element.setup_cost || '';
    document.getElementById('prop-holding-cost').value = element.holding_cost || '';
    const pitchEl = document.getElementById('prop-pitch');
    if (pitchEl) pitchEl.value = element.pitch || '';
    const epeiEl = document.getElementById('prop-epei');
    if (epeiEl) epeiEl.value = element.epei || '';
    document.getElementById('prop-scrap').value = element.scrap_rate || '';
    document.getElementById('prop-available').value = element.available_time || '';
    document.getElementById('prop-shifts').value = element.shifts || '';

    selectedElement = element;
}

function updateTimeUnitLabels(unit) {
    const label = unit === 'hr' ? 'hrs' : unit === 'min' ? 'min' : 'sec';
    const ctLabel = document.getElementById('prop-ct-label');
    const coLabel = document.getElementById('prop-co-label');
    if (ctLabel) ctLabel.textContent = `C/T (${label})`;
    if (coLabel) coLabel.textContent = `C/O (${label})`;
}

// Hook up the time unit selector
document.getElementById('prop-time-unit')?.addEventListener('change', function() {
    const unit = this.value;
    updateTimeUnitLabels(unit);
    // Re-display current values in new unit
    if (selectedElement) {
        const divisor = unit === 'hr' ? 3600 : unit === 'min' ? 60 : 1;
        document.getElementById('prop-cycle-time').value = selectedElement.cycle_time ? (selectedElement.cycle_time / divisor) : '';
        document.getElementById('prop-changeover').value = selectedElement.changeover_time ? (selectedElement.changeover_time / divisor) : '';
    }
});

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

    // Auto-size supermarket or FIFO
    _fetchPullSizing(inv);
}

function _fetchPullSizing(inv) {
    const el = document.getElementById('pull-sizing-panel');
    if (!el) return;
    const dt = inv.delay_type || 'inventory';
    if (dt !== 'supermarket' && dt !== 'fifo') {
        el.innerHTML = '';
        return;
    }
    if (!currentVSM || !inv.id) { el.innerHTML = ''; return; }

    const endpoint = dt === 'supermarket'
        ? `/api/vsm/${currentVSM.id}/size-supermarket/${inv.id}/`
        : `/api/vsm/${currentVSM.id}/size-fifo/${inv.id}/`;

    el.innerHTML = '<div style="font-size:0.7rem; color:var(--text-dim); padding:4px 0;">Sizing...</div>';
    fetch(endpoint, { credentials: 'include' })
        .then(r => r.ok ? r.json() : r.json().then(e => { throw e; }))
        .then(data => {
            let html = '';
            if (dt === 'supermarket') {
                html += `<div class="smp-section-title" style="margin-top:8px;">Supermarket Sizing</div>`;
                if (data.upstream_step || data.downstream_step) {
                    html += `<div style="font-size:0.65rem; color:var(--text-dim); margin-bottom:4px;">${data.upstream_step || '(supplier)'} → supermarket → ${data.downstream_step || '?'}</div>`;
                }
                html += `<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:4px; margin-bottom:6px;">`;
                html += `<div class="smp-kpi"><div class="smp-kpi-label">Kanban Cards</div><div class="smp-kpi-value">${data.kanban_cards || '-'}</div></div>`;
                html += `<div class="smp-kpi"><div class="smp-kpi-label">Container</div><div class="smp-kpi-value">${data.container_size || '-'}</div></div>`;
                html += `<div class="smp-kpi"><div class="smp-kpi-label">Total Units</div><div class="smp-kpi-value">${data.total_units || '-'}</div></div>`;
                html += `</div>`;
                html += `<div style="display:grid; grid-template-columns:1fr 1fr; gap:4px; margin-bottom:6px;">`;
                html += `<div class="smp-kpi"><div class="smp-kpi-label">Cycle Stock</div><div class="smp-kpi-value">${data.cycle_stock || '-'}</div></div>`;
                html += `<div class="smp-kpi"><div class="smp-kpi-label">Safety Stock</div><div class="smp-kpi-value">${data.safety_stock || '-'}</div></div>`;
                html += `</div>`;
                html += `<div style="display:grid; grid-template-columns:1fr 1fr; gap:4px; margin-bottom:6px;">`;
                html += `<div class="smp-kpi"><div class="smp-kpi-label">Avg WIP</div><div class="smp-kpi-value">${data.avg_wip_units || '-'}</div></div>`;
                html += `<div class="smp-kpi"><div class="smp-kpi-label">Shelf Max</div><div class="smp-kpi-value">${data.max_units || '-'}</div></div>`;
                html += `</div>`;
                if (data.holding_cost_per_day > 0) {
                    html += `<div style="font-size:0.65rem; color:var(--text-dim);">Holding: $${(data.holding_cost_per_day || 0).toFixed(2)}/day ($${((data.holding_cost_per_day || 0) * 260).toFixed(0)}/yr)</div>`;
                }
                html += `<div style="font-size:0.7rem; color:var(--text-secondary); margin-top:4px;">${data.reasoning || ''}</div>`;
            } else {
                html += `<div class="smp-section-title" style="margin-top:8px;">FIFO Lane Sizing</div>`;
                if (data.upstream_step && data.downstream_step) {
                    html += `<div style="font-size:0.65rem; color:var(--text-dim); margin-bottom:4px;">${data.upstream_step} → FIFO → ${data.downstream_step}</div>`;
                }
                html += `<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:4px; margin-bottom:6px;">`;
                html += `<div class="smp-kpi"><div class="smp-kpi-label">Max Units</div><div class="smp-kpi-value">${data.max_units || '-'}</div></div>`;
                html += `<div class="smp-kpi"><div class="smp-kpi-label">Avg WIP</div><div class="smp-kpi-value">${data.avg_wip || '-'}</div></div>`;
                html += `<div class="smp-kpi"><div class="smp-kpi-label">Rate Ratio</div><div class="smp-kpi-value">${data.rate_mismatch || '-'}×</div></div>`;
                html += `</div>`;
                if (data.blocking_risk && data.blocking_risk !== 'none') {
                    const riskColor = data.blocking_risk === 'high' ? '#e05252' : data.blocking_risk === 'moderate' ? '#e89547' : 'var(--text-dim)';
                    html += `<div style="font-size:0.7rem; color:${riskColor}; padding:4px 8px; background:rgba(224,82,82,0.08); border-radius:4px; margin-bottom:6px;">Blocking risk: <strong>${data.blocking_risk}</strong>${data.fill_time_sec ? ` — fills in ${Math.round(data.fill_time_sec)}s if downstream stops` : ''}</div>`;
                }
                html += `<div style="font-size:0.7rem; color:var(--text-secondary); margin-top:4px;">${data.reasoning || ''}</div>`;
            }
            el.innerHTML = html;
        })
        .catch(err => {
            const msg = err.error || err.message || 'Could not auto-size';
            el.innerHTML = `<div style="font-size:0.7rem; color:var(--text-dim); padding:4px 0;">${msg}</div>`;
        });
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
        // Save process properties — convert time inputs to seconds (canonical unit)
        const tuSel = document.getElementById('prop-time-unit');
        const timeUnit = tuSel ? tuSel.value : 'sec';
        const multiplier = timeUnit === 'hr' ? 3600 : timeUnit === 'min' ? 60 : 1;

        selectedElement.name = document.getElementById('prop-name').value;
        const ctVal = parseFloat(document.getElementById('prop-cycle-time').value);
        selectedElement.cycle_time = ctVal ? Math.round(ctVal * multiplier * 100) / 100 : null;
        const coVal = parseFloat(document.getElementById('prop-changeover').value);
        selectedElement.changeover_time = coVal ? Math.round(coVal * multiplier * 100) / 100 : null;
        selectedElement.uptime = parseFloat(document.getElementById('prop-uptime').value) || null;
        selectedElement.operators = parseInt(document.getElementById('prop-operators').value) || null;
        selectedElement.demand_rate = parseFloat(document.getElementById('prop-demand-rate').value) || null;
        selectedElement.demand_unit = document.getElementById('prop-demand-unit').value || '';
        selectedElement.batch_size = parseInt(document.getElementById('prop-batch').value) || null;
        selectedElement.batch_process = document.getElementById('prop-batch-process').checked;
        selectedElement.unit_cost = parseFloat(document.getElementById('prop-unit-cost').value) || null;
        selectedElement.setup_cost = parseFloat(document.getElementById('prop-setup-cost').value) || null;
        selectedElement.holding_cost = parseFloat(document.getElementById('prop-holding-cost').value) || null;
        const pitchSave = document.getElementById('prop-pitch');
        if (pitchSave) selectedElement.pitch = parseFloat(pitchSave.value) || null;
        const epeiSave = document.getElementById('prop-epei');
        if (epeiSave) selectedElement.epei = parseFloat(epeiSave.value) || null;
        selectedElement.scrap_rate = parseFloat(document.getElementById('prop-scrap').value) || null;
        selectedElement.available_time = parseInt(document.getElementById('prop-available').value) || null;
        selectedElement.shifts = parseInt(document.getElementById('prop-shifts').value) || null;
    }

    document.getElementById('properties-panel').classList.remove('visible');
    renderVSM();
    updateMetrics();
    saveVSM();
}

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
function updateMetrics() {
    if (!currentVSM) return;

    // Calculate totals (work center aware — parallel machines use effective CT)
    let totalCT = 0;
    let totalWait = 0;

    const wcSteps = {};
    (currentVSM.process_steps || []).forEach(step => {
        const ct = step.cycle_time || 0;
        if (step.work_center_id) {
            if (!wcSteps[step.work_center_id]) wcSteps[step.work_center_id] = [];
            wcSteps[step.work_center_id].push(ct);
        } else {
            totalCT += ct;
        }
    });
    // Add effective CT for each work center
    Object.values(wcSteps).forEach(cts => {
        const rateSum = cts.reduce((s, ct) => ct > 0 ? s + 1.0/ct : s, 0);
        if (rateSum > 0) totalCT += 1.0 / rateSum;
    });

    (currentVSM.inventory || []).forEach(inv => {
        totalWait += inv.days_of_supply || inv.computed_days || 0;
    });

    const leadTime = totalWait + (totalCT / 86400);
    const pce = leadTime > 0 ? ((totalCT / 86400) / leadTime * 100) : 0;

    document.getElementById('metric-lead-time').textContent = leadTime.toFixed(2) + ' days';
    document.getElementById('metric-process-time').textContent = totalCT + ' sec';
    document.getElementById('metric-pce').textContent = pce.toFixed(1) + '%';
    document.getElementById('metric-takt').textContent = currentVSM.takt_time ? currentVSM.takt_time + ' sec' : '-';

    // Pre-fill takt inputs if data exists
    if (currentVSM.takt_time) {
        document.getElementById('takt-direct').value = currentVSM.takt_time;
    }

    // Client-side bottleneck detection
    const bn = detectBottleneckClient(currentVSM);
    const bnEl = document.getElementById('metric-bottleneck');
    const tpEl = document.getElementById('metric-throughput');
    if (bn) {
        bnEl.textContent = bn.name + ' (' + formatTime(bn.ct) + ')';
        bnEl.style.color = (currentVSM.takt_time && bn.ct > currentVSM.takt_time) ? '#e74c3c' : '';
        tpEl.textContent = bn.throughput.toFixed(1) + ' u/hr';
    } else {
        bnEl.textContent = '-';
        bnEl.style.color = '';
        tpEl.textContent = '-';
    }

    // Suggested calculators
    renderSuggestedCalcs(bn);
}

function detectBottleneckClient(vsm) {
    // Client-side bottleneck detection — mirrors server logic
    const steps = vsm.process_steps || [];
    const wcs = vsm.work_centers || [];
    if (steps.length === 0) return null;

    const wcMap = {};
    wcs.forEach(wc => { wcMap[wc.id] = { ...wc, members: [] }; });
    const standalone = [];
    steps.forEach(s => {
        if (s.work_center_id && wcMap[s.work_center_id]) {
            wcMap[s.work_center_id].members.push(s);
        } else {
            standalone.push(s);
        }
    });

    const effective = [];
    standalone.forEach(s => {
        const ct = s.cycle_time || 0;
        if (ct > 0) effective.push({ name: s.name || 'Process', ct, id: s.id });
    });
    Object.values(wcMap).forEach(wc => {
        if (wc.members.length === 0) return;
        const rateSum = wc.members.reduce((sum, m) => {
            const ct = m.cycle_time || 0;
            return ct > 0 ? sum + 1.0 / ct : sum;
        }, 0);
        if (rateSum > 0) {
            effective.push({ name: wc.name || wc.members[0].name, ct: 1.0 / rateSum, id: wc.members[0].id });
        }
    });

    if (effective.length === 0) return null;
    const maxCT = Math.max(...effective.map(e => e.ct));
    const bn = effective.find(e => e.ct === maxCT);
    return { name: bn.name, ct: maxCT, throughput: 3600.0 / maxCT, id: bn.id };
}

function renderSuggestedCalcs(bottleneck) {
    const container = document.getElementById('suggested-calcs');
    const list = document.getElementById('suggested-calcs-list');
    const suggestions = [];

    if (!currentVSM || (currentVSM.process_steps || []).length === 0) {
        container.style.display = 'none';
        return;
    }

    if (!currentVSM.takt_time) {
        suggestions.push({ name: 'Takt Time', id: 'takt', reason: 'Set takt to flag overloaded steps' });
    }
    if (bottleneck) {
        if (currentVSM.takt_time && bottleneck.ct > currentVSM.takt_time) {
            suggestions.push({ name: 'SMED', id: 'smed', reason: 'Reduce changeover at bottleneck' });
            suggestions.push({ name: 'Line Simulator', id: 'line-sim', reason: 'Model flow impact' });
        } else {
            suggestions.push({ name: 'Bottleneck Analysis', id: 'bottleneck', reason: 'Detailed constraint analysis' });
        }
    }
    if ((currentVSM.process_steps || []).some(s => !s.uptime || s.uptime < 100)) {
        suggestions.push({ name: 'OEE', id: 'oee', reason: 'Measure equipment effectiveness' });
    }

    if (suggestions.length === 0) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'block';
    list.innerHTML = suggestions.slice(0, 3).map(s =>
        `<a href="/app/calculators/" target="_blank" style="display:block; padding:0.3rem 0.4rem; margin-bottom:0.25rem; background:var(--bg-tertiary); border-radius:4px; text-decoration:none; font-size:0.75rem; color:var(--accent-primary);" title="${s.reason}">
            ${s.name} <span style="color:var(--text-dim); font-size:0.65rem;">— ${s.reason}</span>
        </a>`
    ).join('');
}

// --- Step Metrics Overlay ---
let stepMetricsClickTimer = null;

function showStepMetrics(step) {
    const panel = document.getElementById('step-metrics-panel');
    document.getElementById('smp-title').textContent = step.name || 'Unnamed Step';

    // Banner: bottleneck / exceeds takt / normal
    const flags = step.flags || {};
    const bannerEl = document.getElementById('smp-banner');
    if (flags.is_bottleneck) {
        bannerEl.innerHTML = '<div class="smp-banner red">System Constraint (Bottleneck)</div>';
    } else if (flags.exceeds_takt) {
        bannerEl.innerHTML = '<div class="smp-banner yellow">Exceeds Takt — ratio ' + (flags.takt_ratio || '?') + '</div>';
    } else if (flags.takt_ratio) {
        bannerEl.innerHTML = '<div class="smp-banner green">Within Takt — ratio ' + flags.takt_ratio + '</div>';
    } else {
        bannerEl.innerHTML = '';
    }

    // KPI grid — auto-format time values
    const fmtT = (s) => !s ? '-' : s < 60 ? s + 's' : s < 3600 ? (s/60).toFixed(1) + 'm' : (s/3600).toFixed(2) + 'h';
    const ct = step.cycle_time ? fmtT(step.cycle_time) : '-';
    const co = step.changeover_time ? fmtT(step.changeover_time) : '-';
    const uptime = step.uptime ? step.uptime + '%' : '-';
    const ops = step.operators || '-';
    const batch = step.batch_size || '-';
    const pitchVal = step.pitch ? step.pitch + 'm' : '-';
    const epeiVal = step.epei ? step.epei + 'd' : '-';
    const taktRatio = flags.takt_ratio || '-';
    document.getElementById('smp-kpis').innerHTML =
        `<div class="smp-kpi"><div class="smp-kpi-label">C/T</div><div class="smp-kpi-value">${ct}</div></div>` +
        `<div class="smp-kpi"><div class="smp-kpi-label">C/O</div><div class="smp-kpi-value">${co}</div></div>` +
        `<div class="smp-kpi"><div class="smp-kpi-label">Uptime</div><div class="smp-kpi-value">${uptime}</div></div>` +
        `<div class="smp-kpi"><div class="smp-kpi-label">Operators</div><div class="smp-kpi-value">${ops}</div></div>` +
        `<div class="smp-kpi"><div class="smp-kpi-label">Batch</div><div class="smp-kpi-value">${batch}</div></div>` +
        `<div class="smp-kpi"><div class="smp-kpi-label">Pitch</div><div class="smp-kpi-value">${pitchVal}</div></div>` +
        `<div class="smp-kpi"><div class="smp-kpi-label">EPEI</div><div class="smp-kpi-value">${epeiVal}</div></div>` +
        `<div class="smp-kpi"><div class="smp-kpi-label">vs Takt</div><div class="smp-kpi-value">${taktRatio}</div></div>`;

    // Lot size recommendation + counterfactual analysis (async fetch)
    const lotEl = document.getElementById('smp-lot-rec');
    if (lotEl && currentVSM && step.id) {
        lotEl.innerHTML = '<div style="font-size:0.7rem; color:var(--text-dim); padding:4px 0;">Analyzing...</div>';
        const base = `/api/vsm/${currentVSM.id}`;
        const sid = step.id;

        // Fetch lot rec, SMED impact, and EPEI in parallel
        Promise.all([
            fetch(`${base}/lot-recommendation/${sid}/`, { credentials: 'include' }).then(r => r.ok ? r.json() : null),
            fetch(`${base}/smed-impact/${sid}/`, { credentials: 'include' }).then(r => r.ok ? r.json() : null).catch(() => null),
            fetch(`${base}/epei/${sid}/`, { credentials: 'include' }).then(r => r.ok ? r.json() : null).catch(() => null),
        ]).then(([data, smedData, epeiData]) => {
                if (!data) { lotEl.innerHTML = ''; return; }
                const rec = data.recommendation || {};
                const regime = data.regime || {};
                const kanban = data.kanban || {};
                const regimeLabel = (regime.regime || '').replace(/_/g, ' ');
                const conf = regime.confidence ? `${Math.round(regime.confidence * 100)}%` : '';

                let html = `<div class="smp-section-title" style="margin-top:8px;">Lot Size Recommendation</div>`;
                html += `<div style="font-size:0.65rem; color:var(--accent); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">${regimeLabel} (${conf} confidence)</div>`;

                // Feasibility warning
                const scenarios = rec.scenarios || [];
                const bestScenario = rec.best_scenario || {};
                const bestUtil = bestScenario.equipment_utilization_pct || bestScenario.changeover_pct || 0;
                if (rec.feasible === false || bestUtil > 100) {
                    html += `<div style="font-size:0.7rem; color:#e05252; padding:6px 8px; background:rgba(224,82,82,0.1); border-radius:4px; margin-bottom:6px; border-left:3px solid #e05252;">`;
                    html += `<strong>Infeasible at recommended lot.</strong> `;
                    if (rec.smed_target) {
                        const st = rec.smed_target;
                        html += `Changeover must shrink from ${st.current_co_min} min to ${st.required_co_min} min (${st.reduction_pct}% SMED reduction) to make this work.`;
                    } else {
                        html += `Utilization exceeds available capacity. Consider parallel stations, overtime, or SMED to reduce changeover burden.`;
                    }
                    html += `</div>`;
                }

                // Takt vs CT assessment
                if (rec.takt_vs_ct) {
                    const tvc = rec.takt_vs_ct;
                    html += `<div style="font-size:0.75rem; padding:6px 8px; background:var(--bg-primary); border-radius:4px; margin-bottom:6px;">`;
                    html += `<div><strong>Takt:</strong> ${tvc.takt_display} | <strong>C/T:</strong> ${tvc.ct_display}`;
                    if (tvc.ratio) html += ` | <strong>Ratio:</strong> ${tvc.ratio}`;
                    html += `</div>`;
                    html += `<div style="margin-top:4px; color:var(--text-secondary);">${tvc.assessment}</div>`;
                    html += `</div>`;
                }

                // Lot size KPIs
                html += `<div style="display:grid; grid-template-columns:1fr 1fr; gap:4px; margin-bottom:6px;">`;
                html += `<div class="smp-kpi"><div class="smp-kpi-label">Recommended Lot</div><div class="smp-kpi-value">${rec.lot_size ?? '-'}</div></div>`;
                if (rec.epei_days) {
                    html += `<div class="smp-kpi"><div class="smp-kpi-label">EPEI</div><div class="smp-kpi-value">${rec.epei_days}d</div></div>`;
                } else if (rec.epei) {
                    html += `<div class="smp-kpi"><div class="smp-kpi-label">EPEI</div><div class="smp-kpi-value">${rec.epei}</div></div>`;
                }
                html += `</div>`;

                // Kanban sizing
                if (kanban.kanban_cards) {
                    html += `<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:4px; margin-bottom:6px;">`;
                    html += `<div class="smp-kpi"><div class="smp-kpi-label">Kanban Cards</div><div class="smp-kpi-value">${kanban.kanban_cards}</div></div>`;
                    html += `<div class="smp-kpi"><div class="smp-kpi-label">Container</div><div class="smp-kpi-value">${kanban.container_size}</div></div>`;
                    html += `<div class="smp-kpi"><div class="smp-kpi-label">In Loop</div><div class="smp-kpi-value">${kanban.total_units_in_loop}</div></div>`;
                    html += `</div>`;
                    html += `<div style="font-size:0.7rem; color:var(--text-dim); margin-bottom:6px;">${kanban.reasoning}</div>`;
                }

                // Batch warning
                if (rec.batch_warning) {
                    html += `<div style="font-size:0.7rem; color:#e89547; padding:4px 8px; background:rgba(232,149,71,0.1); border-radius:4px; margin-bottom:6px;">${rec.batch_warning}</div>`;
                }

                // Cost basis
                const cb = rec.cost_basis;
                if (cb) {
                    const tag = cb.estimated ? ' (estimated — enter costs for precision)' : '';
                    html += `<div style="font-size:0.65rem; color:var(--text-dim); margin-top:4px;">Cost basis: setup $${cb.setup_cost_per_changeover}/changeover, holding $${cb.holding_cost_per_unit_day}/unit/day${tag}</div>`;
                }

                // Reasoning
                html += `<div style="font-size:0.7rem; color:var(--text-secondary); margin-top:4px;">${rec.reasoning || ''}</div>`;

                // ============================================================
                // COUNTERFACTUAL: SMED What-If Slider
                // ============================================================
                if (smedData && smedData.results && smedData.results.length > 1 && smedData.results[0].daily_cost !== undefined) {
                    const smed = smedData.results;
                    const baseline = smed[0];
                    html += `<div style="margin-top:12px; padding:8px; background:var(--bg-primary); border-radius:6px; border-left:3px solid var(--accent);">`;
                    html += `<div class="smp-section-title" style="margin:0 0 6px;">What if you reduced changeover?</div>`;
                    html += `<div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">`;
                    html += `<input type="range" id="smed-slider" min="0" max="${smed.length - 1}" value="0" style="flex:1; accent-color:var(--accent);">`;
                    html += `<span id="smed-pct" style="font-size:0.75rem; font-weight:600; min-width:32px;">0%</span>`;
                    html += `</div>`;
                    html += `<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:4px;" id="smed-kpis">`;
                    html += `<div class="smp-kpi"><div class="smp-kpi-label">Changeover</div><div class="smp-kpi-value" id="smed-co">${baseline.changeover_display}</div></div>`;
                    html += `<div class="smp-kpi"><div class="smp-kpi-label">Lot Size</div><div class="smp-kpi-value" id="smed-lot">${baseline.lot_size}</div></div>`;
                    html += `<div class="smp-kpi"><div class="smp-kpi-label">Avg WIP</div><div class="smp-kpi-value" id="smed-wip">${baseline.avg_wip}</div></div>`;
                    html += `</div>`;
                    html += `<div style="display:grid; grid-template-columns:1fr 1fr; gap:4px; margin-top:4px;">`;
                    html += `<div class="smp-kpi"><div class="smp-kpi-label">Daily Cost</div><div class="smp-kpi-value" id="smed-cost">$${(baseline.daily_cost || 0).toFixed(2)}</div></div>`;
                    html += `<div class="smp-kpi"><div class="smp-kpi-label">Savings/Day</div><div class="smp-kpi-value" id="smed-save" style="color:var(--text-dim);">—</div></div>`;
                    html += `</div>`;
                    html += `<div id="smed-annual" style="font-size:0.7rem; color:var(--text-dim); margin-top:4px; text-align:center;"></div>`;
                    html += `</div>`;

                    // Wire slider after innerHTML is set — deferred
                    setTimeout(() => {
                        const slider = document.getElementById('smed-slider');
                        if (!slider) return;
                        slider.addEventListener('input', () => {
                            const i = parseInt(slider.value);
                            const s = smed[i];
                            document.getElementById('smed-pct').textContent = `${s.reduction_pct}%`;
                            document.getElementById('smed-co').textContent = s.changeover_display;
                            document.getElementById('smed-lot').textContent = s.lot_size;
                            document.getElementById('smed-wip').textContent = s.avg_wip;
                            document.getElementById('smed-cost').textContent = `$${s.daily_cost.toFixed(2)}`;
                            if (s.cost_savings_per_day && s.cost_savings_per_day > 0) {
                                const annual = (s.cost_savings_per_day * 260);
                                document.getElementById('smed-save').textContent = `$${s.cost_savings_per_day.toFixed(2)}`;
                                document.getElementById('smed-save').style.color = 'var(--accent)';
                                document.getElementById('smed-annual').textContent = `$${annual.toLocaleString('en-US', {maximumFractionDigits:0})}/yr in lot size cost alone`;
                                document.getElementById('smed-annual').style.color = 'var(--accent)';
                            } else {
                                document.getElementById('smed-save').textContent = '—';
                                document.getElementById('smed-save').style.color = 'var(--text-dim)';
                                document.getElementById('smed-annual').textContent = '';
                            }
                        });
                    }, 0);
                }

                // ============================================================
                // COUNTERFACTUAL: EPEI Cycling Options
                // ============================================================
                if (epeiData && epeiData.options && epeiData.options.length > 0) {
                    const opts = epeiData.options;
                    html += `<details style="margin-top:8px;">`;
                    html += `<summary style="cursor:pointer; font-size:0.75rem; font-weight:600; color:var(--text-secondary);">EPEI Cycling Options</summary>`;
                    html += `<div style="margin-top:6px;">`;
                    html += `<table style="width:100%; font-size:0.65rem; border-collapse:collapse;">`;
                    html += `<tr style="color:var(--text-dim); border-bottom:1px solid var(--border);">`;
                    html += `<th style="text-align:left; padding:3px 4px;">Cycle</th>`;
                    html += `<th style="text-align:right; padding:3px 4px;">Lot</th>`;
                    html += `<th style="text-align:right; padding:3px 4px;">CO %</th>`;
                    html += `<th style="text-align:right; padding:3px 4px;">Util %</th>`;
                    if (opts[0].daily_cost !== undefined) html += `<th style="text-align:right; padding:3px 4px;">$/day</th>`;
                    html += `<th style="text-align:center; padding:3px 4px;"></th>`;
                    html += `</tr>`;
                    for (const o of opts) {
                        const rowColor = o.feasible ? 'inherit' : 'rgba(224,82,82,0.15)';
                        const fIcon = o.feasible ? '<span style="color:var(--accent);">&#10003;</span>' : '<span style="color:#e05252;">&#10007;</span>';
                        html += `<tr style="background:${rowColor}; border-bottom:1px solid var(--border);">`;
                        html += `<td style="padding:3px 4px;">${o.label}</td>`;
                        html += `<td style="text-align:right; padding:3px 4px;">${o.lot_size_per_part}</td>`;
                        html += `<td style="text-align:right; padding:3px 4px;">${o.changeover_pct}%</td>`;
                        html += `<td style="text-align:right; padding:3px 4px;">${o.utilization_pct}%</td>`;
                        if (o.daily_cost !== undefined) html += `<td style="text-align:right; padding:3px 4px;">$${(o.daily_cost || 0).toFixed(2)}</td>`;
                        html += `<td style="text-align:center; padding:3px 4px;">${fIcon}</td>`;
                        html += `</tr>`;
                        if (!o.feasible && o.smed_target_min) {
                            html += `<tr style="background:rgba(224,82,82,0.08);">`;
                            const cols = o.daily_cost !== undefined ? 6 : 5;
                            html += `<td colspan="${cols}" style="padding:2px 4px 4px; font-size:0.6rem; color:#e89547;">SMED target: ${o.smed_target_min} min (${o.smed_reduction_pct}% reduction) to enable ${o.label} cycling</td>`;
                            html += `</tr>`;
                        }
                    }
                    html += `</table>`;
                    if (opts.every(o => !o.feasible)) {
                        html += `<div style="font-size:0.65rem; color:#e05252; padding:4px 8px; margin-top:4px; background:rgba(224,82,82,0.08); border-radius:4px;">No cycling frequency is feasible at current changeover time. SMED is required before implementing any EPEI target.</div>`;
                    }
                    html += `</div></details>`;
                }

                // Scenario table (existing scenarios from lot rec)
                if (scenarios.length > 1) {
                    html += `<details style="margin-top:8px;">`;
                    html += `<summary style="cursor:pointer; font-size:0.75rem; font-weight:600; color:var(--text-secondary);">Cost Scenarios</summary>`;
                    html += `<div style="margin-top:6px;">`;
                    html += `<table style="width:100%; font-size:0.65rem; border-collapse:collapse;">`;
                    html += `<tr style="color:var(--text-dim); border-bottom:1px solid var(--border);">`;
                    html += `<th style="text-align:left; padding:3px 4px;">Lot</th>`;
                    html += `<th style="text-align:right; padding:3px 4px;">DOS</th>`;
                    html += `<th style="text-align:right; padding:3px 4px;">WIP</th>`;
                    html += `<th style="text-align:right; padding:3px 4px;">$/day</th>`;
                    html += `</tr>`;
                    for (const s of scenarios) {
                        const mark = s.is_epq ? ' (EPQ)' : s.is_current ? ' (now)' : '';
                        const isBest = rec.best_scenario && s.lot_size === rec.best_scenario.lot_size;
                        const rowStyle = isBest ? 'font-weight:600; color:var(--accent);' : '';
                        html += `<tr style="border-bottom:1px solid var(--border); ${rowStyle}">`;
                        html += `<td style="padding:3px 4px;">${s.lot_size}${mark}</td>`;
                        html += `<td style="text-align:right; padding:3px 4px;">${s.days_of_supply}d</td>`;
                        html += `<td style="text-align:right; padding:3px 4px;">${s.avg_wip}</td>`;
                        html += `<td style="text-align:right; padding:3px 4px;">$${(s.daily_cost || 0).toFixed(2)}</td>`;
                        html += `</tr>`;
                    }
                    html += `</table>`;
                    html += `</div></details>`;
                }

                // Methodology
                html += `<details style="margin-top:8px; font-size:0.65rem; color:var(--text-dim);">`;
                html += `<summary style="cursor:pointer; color:var(--text-secondary);">How this is calculated</summary>`;
                html += `<div style="margin-top:6px; line-height:1.5; padding:6px 8px; background:var(--bg-primary); border-radius:4px;">`;
                html += `<p style="margin:0 0 6px;"><strong>Priority order:</strong> Customer demand → Cost optimization → Equipment constraints.</p>`;
                html += `<p style="margin:0 0 6px;"><strong>1. Regime detection</strong> classifies the operating environment — schedule-paced (low demand), mix-constrained (changeover burden from product variety), or capacity-budget (running near capacity). This determines context, not the formula.</p>`;
                html += `<p style="margin:0 0 6px;"><strong>2. Cost analysis</strong> uses the Economic Production Quantity (EPQ) model: lot size that minimizes the sum of setup cost (per changeover) and holding cost (per unit per day of average WIP). The same formula runs in every regime — no discontinuity at boundaries.</p>`;
                html += `<p style="margin:0 0 6px;"><strong>3. Demand ceiling</strong> caps the recommendation. The cost-optimal lot may suggest months of inventory, but the recommendation never exceeds a reasonable supply horizon (1–5 days depending on demand rate). The EPQ reference is shown for comparison when demand overrides cost.</p>`;
                html += `<p style="margin:0 0 6px;"><strong>4. Batch process override.</strong> When a step is marked as a batch process (oven, furnace, tank), the equipment capacity sets the lot size floor. Scenario analysis compares full vs partial batches at different run frequencies.</p>`;
                if (cb && cb.estimated) {
                    html += `<p style="margin:0; color:#e89547;"><strong>Cost inputs are estimated</strong> from cycle time and changeover duration. Enter Unit Cost, Setup Cost, or Holding Cost in the step properties for precision — estimated costs can shift the optimal lot by 4× or more.</p>`;
                }
                html += `</div></details>`;

                lotEl.innerHTML = html;
            })
            .catch(() => { lotEl.innerHTML = ''; });
    }

    // Annotations
    const annotations = step.annotations || [];
    const annEl = document.getElementById('smp-annotations');
    if (annotations.length > 0) {
        annEl.innerHTML = '<div class="smp-section-title">Linked Results</div>' +
            annotations.map((a, i) => renderAnnotationCard(a, step, i)).join('');
    } else {
        annEl.innerHTML = '';
    }

    panel.classList.add('visible');
}

function renderAnnotationCard(a, step, idx) {
    const statusClass = a.status || '';
    const source = (a.source || 'unknown').replace(/_/g, ' ');
    const ts = a.timestamp ? new Date(a.timestamp).toLocaleDateString() : '';
    let body = '';
    if (a.value && typeof a.value === 'object') {
        body = Object.entries(a.value).map(([k, v]) =>
            `<span style="margin-right:8px;">${k.replace(/_/g, ' ')}: <strong>${v}</strong></span>`
        ).join('');
    } else if (a.value) {
        body = String(a.value);
    }
    const stepId = step.id || '';
    return `<div class="smp-annotation-card ${statusClass}">
        <button class="smp-ann-remove" onclick="removeAnnotation('${stepId}', ${idx})" title="Remove">&times;</button>
        <div class="smp-ann-source">${source}</div>
        <div class="smp-ann-body">${body}</div>
        ${ts ? '<div class="smp-ann-time">' + ts + '</div>' : ''}
    </div>`;
}

function removeAnnotation(stepId, idx) {
    if (!currentVSM) return;
    const step = (currentVSM.process_steps || []).find(s => s.id === stepId);
    if (!step || !step.annotations) return;
    saveVSMState();
    step.annotations.splice(idx, 1);
    showStepMetrics(step);
    renderVSM();
    saveVSM();
}

function closeStepMetrics() {
    document.getElementById('step-metrics-panel').classList.remove('visible');
}

// --- Hypothesis-driven kaizen tracking (Phase 5) ---

async function loadKaizenHypotheses(burst) {
    const select = document.getElementById('prop-kaizen-hypothesis');
    const probEl = document.getElementById('prop-kaizen-prob');
    select.innerHTML = '<option value="">— None —</option>';
    probEl.style.display = 'none';

    if (!currentVSM || !currentVSM.project_id) {
        select.innerHTML = '<option value="">— Link VSM to project first —</option>';
        return;
    }
    try {
        const resp = await fetch(`/api/synara/${currentVSM.project_id}/hypotheses/`, { credentials: 'same-origin' });
        if (!resp.ok) return;
        const data = await resp.json();
        const hypotheses = data.hypotheses || [];
        for (const h of hypotheses) {
            const opt = document.createElement('option');
            opt.value = h.id;
            opt.textContent = `${h.description.slice(0, 40)}${h.description.length > 40 ? '...' : ''} (P=${Math.round((h.posterior || h.prior || 0.5) * 100)}%)`;
            if (burst.hypothesis_id === h.id) opt.selected = true;
            select.appendChild(opt);
        }
        if (burst.hypothesis_id) {
            const linked = hypotheses.find(h => h.id === burst.hypothesis_id);
            if (linked) {
                const prob = linked.posterior || linked.prior || 0.5;
                probEl.textContent = `Current P(H) = ${Math.round(prob * 100)}%`;
                probEl.style.display = 'block';
            }
        }
    } catch (e) {
        // Silently fail — hypothesis linking is optional
    }
}

async function createHypothesisFromKaizen() {
    if (!currentVSM || !currentVSM.project_id) {
        alert('Link this VSM to a project first (in VSM properties).');
        return;
    }
    if (!selectedElement || selectedElementType !== 'kaizen') return;

    const burstText = selectedElement.text || 'improvement';
    // Find nearest process step
    let nearestStep = 'the process';
    let minDist = Infinity;
    for (const step of (currentVSM.process_steps || [])) {
        const dx = (step.x || 0) - (selectedElement.x || 0);
        const dy = (step.y || 0) - (selectedElement.y || 0);
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < minDist) { minDist = dist; nearestStep = step.name || 'this step'; }
    }

    const description = `If we ${burstText.toLowerCase()} at ${nearestStep}, then lead time decreases and throughput improves`;
    try {
        const resp = await fetch(`/api/synara/${currentVSM.project_id}/hypotheses/add/`, {
            method: 'POST', credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                description: description,
                behavior_class: 'improvement',
                prior: 0.5
            })
        });
        if (!resp.ok) throw new Error('Failed to create hypothesis');
        const data = await resp.json();
        const h = data.hypothesis;
        selectedElement.hypothesis_id = h.id;
        await loadKaizenHypotheses(selectedElement);
        document.getElementById('prop-kaizen-hypothesis').value = h.id;
        renderVSM();
        saveVSM();
    } catch (e) {
        alert('Failed to create hypothesis: ' + safeStr(e, 'Unknown error'));
    }
}

async function loadHypothesisProbabilities() {
    // Called after loading a VSM — enrich kaizen bursts with hypothesis probabilities
    if (!currentVSM || !currentVSM.project_id) return;
    const bursts = currentVSM.kaizen_bursts || [];
    const linked = bursts.filter(b => b.hypothesis_id);
    if (linked.length === 0) return;
    try {
        const resp = await fetch(`/api/synara/${currentVSM.project_id}/hypotheses/`, { credentials: 'same-origin' });
        if (!resp.ok) return;
        const data = await resp.json();
        const hMap = {};
        for (const h of (data.hypotheses || [])) { hMap[h.id] = h.posterior || h.prior || 0.5; }
        for (const b of linked) {
            b._hypothesis_prob = hMap[b.hypothesis_id] || null;
        }
    } catch (e) { /* optional enrichment */ }
}

function showTimeline() {
    if (!currentVSM) return;
    const snaps = currentVSM.metric_snapshots || [];
    const modal = document.getElementById('timeline-modal');
    modal.style.display = 'flex';

    if (snaps.length < 2) {
        document.getElementById('timeline-chart').style.display = 'none';
        document.getElementById('timeline-empty').style.display = 'block';
        return;
    }
    document.getElementById('timeline-chart').style.display = 'block';
    document.getElementById('timeline-empty').style.display = 'none';

    const timestamps = snaps.map(s => s.timestamp ? new Date(s.timestamp).toLocaleString() : '');
    const leadTimes = snaps.map(s => s.lead_time || 0);
    const pces = snaps.map(s => s.pce || 0);

    const traces = [
        { x: timestamps, y: leadTimes, name: 'Lead Time (days)', type: 'scatter', mode: 'lines+markers', line: { color: '#e74c3c' } },
        { x: timestamps, y: pces, name: 'PCE (%)', type: 'scatter', mode: 'lines+markers', yaxis: 'y2', line: { color: '#4a9f6e' } }
    ];
    const layout = {
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#999', size: 11 },
        margin: { t: 30, b: 60, l: 60, r: 60 },
        xaxis: { showgrid: false },
        yaxis: { title: 'Lead Time (days)', gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis2: { title: 'PCE (%)', overlaying: 'y', side: 'right', gridcolor: 'rgba(255,255,255,0.05)' },
        legend: { orientation: 'h', y: -0.2 },
        showlegend: true
    };
    Plotly.newPlot('timeline-chart', traces, layout, { responsive: true, displayModeBar: false });
}

function setTaktDirect() {
    const val = parseFloat(document.getElementById('takt-direct').value);
    if (!val || val <= 0 || !currentVSM) return;
    saveVSMState();
    currentVSM.takt_time = Math.round(val * 10) / 10;
    updateMetrics();
    saveVSM();
}

function calculateTakt() {
    const avail = parseFloat(document.getElementById('takt-avail').value);
    const demand = parseFloat(document.getElementById('takt-demand').value);
    if (!avail || !demand || demand <= 0 || !currentVSM) return;
    saveVSMState();
    const takt = Math.round((avail / demand) * 10) / 10;
    currentVSM.takt_time = takt;
    document.getElementById('takt-direct').value = takt;
    updateMetrics();
    saveVSM();
}

async function createFutureState() {
    if (!vsmId) return;

    try {
        const response = await fetch(`/api/vsm/${vsmId}/future-state/`, {
            method: 'POST',
            credentials: 'include'
        });

        if (!response.ok) throw new Error('Failed to create future state');
        const data = await response.json();
        window.location.href = `/app/vsm/${data.future_state.id}/`;
    } catch (err) {
        console.error('Create future state error:', err);
    }
}

async function compareStates() {
    if (!vsmId) return;

    try {
        const response = await fetch(`/api/vsm/${vsmId}/compare/`, { credentials: 'include' });
        const data = await response.json();

        if (data.comparison) {
            alert(`Comparison:\nLead Time: ${data.comparison.lead_time.improvement.toFixed(1)}% improvement\nPCE: ${data.comparison.pce.improvement.toFixed(1)}% improvement`);
        } else {
            alert('No future state found to compare.');
        }
    } catch (err) {
        console.error('Compare error:', err);
    }
}

function exportVSM() {
    // TODO: PDF/image export
    alert('Export coming soon');
}

// =============================================================================
// CI Proposals (Enterprise)
// =============================================================================
let proposalData = [];

function updateProposalButton() {
    const btn = document.getElementById('btn-generate-proposals');
    if (!btn) return;
    const user = window.svendUser;
    if (user && user.features && user.features.hoshin_kanri && currentVSM && currentVSM.status !== 'future') {
        btn.style.display = 'block';
    } else {
        btn.style.display = 'none';
    }
}

// Also check when user data arrives (may happen before or after VSM load)
window.addEventListener('svendUserReady', function() { updateProposalButton(); });

function openProposalModal() {
    const modal = document.getElementById('proposals-modal');
    modal.style.display = 'flex';
    // Reset to params step
    document.getElementById('proposal-params').style.display = 'block';
    document.getElementById('proposal-results').style.display = 'none';
    document.getElementById('proposal-loading').style.display = 'none';
    document.getElementById('proposal-error').style.display = 'none';
}

function closeProposalModal() {
    document.getElementById('proposals-modal').style.display = 'none';
}

async function generateVSMProposals() {
    const volume = document.getElementById('proposal-volume').value || 100000;
    const cost = document.getElementById('proposal-cost').value || 50;

    document.getElementById('proposal-params').style.display = 'none';
    document.getElementById('proposal-loading').style.display = 'block';
    document.getElementById('proposal-error').style.display = 'none';

    try {
        const response = await fetch(`/api/vsm/${vsmId}/generate-proposals/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ annual_volume: parseFloat(volume), cost_per_unit: parseFloat(cost) })
        });

        const data = await response.json();
        if (!response.ok) {
            document.getElementById('proposal-loading').style.display = 'none';
            document.getElementById('proposal-error').style.display = 'block';
            document.getElementById('proposal-error').textContent = safeStr(data.error, 'Failed to generate proposals.');
            document.getElementById('proposal-params').style.display = 'block';
            return;
        }

        proposalData = data.proposals || [];
        document.getElementById('proposal-loading').style.display = 'none';

        if (proposalData.length === 0) {
            document.getElementById('proposal-error').style.display = 'block';
            document.getElementById('proposal-error').textContent = 'No proposals generated. Ensure kaizen bursts exist on the future state.';
            document.getElementById('proposal-params').style.display = 'block';
            return;
        }

        renderProposalCards(proposalData);
        document.getElementById('proposal-results').style.display = 'block';
    } catch (err) {
        document.getElementById('proposal-loading').style.display = 'none';
        document.getElementById('proposal-error').style.display = 'block';
        document.getElementById('proposal-error').textContent = 'Network error: ' + safeStr(err, 'Unknown error');
        document.getElementById('proposal-params').style.display = 'block';
    }
}

function renderProposalCards(proposals) {
    const container = document.getElementById('proposal-list');
    const priorityColors = { high: '#ef4444', medium: '#f59e0b', low: '#6b7280' };

    container.innerHTML = proposals.map((p, i) => `
        <div style="background:var(--bg-tertiary); border:1px solid var(--border); border-radius:4px; padding:1rem; margin-bottom:0.75rem;">
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
                <input type="checkbox" id="proposal-${i}" checked>
                <label for="proposal-${i}" style="font-weight:500; flex:1; cursor:pointer;">${p.suggested_title}</label>
                <span style="font-size:0.7rem; padding:0.15rem 0.5rem; border-radius:2px; background:${priorityColors[p.priority] || '#6b7280'}22; color:${priorityColors[p.priority] || '#6b7280'}; text-transform:uppercase;">${p.priority}</span>
            </div>
            <div style="font-size:0.8rem; color:var(--text-dim); margin-bottom:0.5rem;">
                Process: <strong style="color:var(--text-secondary);">${p.process_step}</strong>
                ${p.has_current_match ? '' : '<span style="color:var(--warning);"> (no baseline match)</span>'}
            </div>
            <div style="display:flex; gap:1rem; font-size:0.75rem; color:var(--text-dim);">
                ${p.metric_deltas.cycle_time ? `<span>C/T: ${p.metric_deltas.cycle_time > 0 ? '-' : '+'}${Math.abs(p.metric_deltas.cycle_time)}s</span>` : ''}
                ${p.metric_deltas.changeover ? `<span>C/O: ${p.metric_deltas.changeover > 0 ? '-' : '+'}${Math.abs(p.metric_deltas.changeover)}s</span>` : ''}
                ${p.metric_deltas.uptime ? `<span>Uptime: ${p.metric_deltas.uptime > 0 ? '+' : ''}${p.metric_deltas.uptime}%</span>` : ''}
            </div>
            <div style="margin-top:0.5rem; font-size:0.85rem;">
                Est. savings: <strong style="color:var(--success);">$${Math.round(p.lower_5 || 0).toLocaleString()} &mdash; $${Math.round(p.upper_95 || 0).toLocaleString()}</strong>/yr
                <span style="font-size:0.7rem; color:var(--text-dim); margin-left:0.25rem;">(90% CI, median $${Math.round(p.median_savings || p.estimated_annual_savings || 0).toLocaleString()})</span>
                ${p.p_positive > 0 ? `<span style="font-size:0.7rem; color:var(--success); margin-left:0.5rem;">${Math.round(p.p_positive * 100)}% chance of positive ROI</span>` : ''}
            </div>
            <button class="sv-btn sv-btn-primary" style="margin-top:0.5rem;font-size:0.75rem;padding:0.3rem 0.8rem;background:var(--accent);border-color:var(--accent);"
                onclick="approveProposal('${p.burst_id||''}', '${(p.suggested_title||'').replace(/'/g,'\\&#39;')}', ${Math.round(p.median_savings || p.estimated_annual_savings || 0)}, '${p.suggested_method||'direct'}', '${p.suggested_type||'material'}')">
                Approve → Hoshin Project
            </button>
        </div>
    `).join('');
}

async function createSelectedProposals() {
    const selected = proposalData.filter((_, i) => document.getElementById(`proposal-${i}`).checked);
    if (selected.length === 0) {
        alert('Select at least one proposal.');
        return;
    }

    try {
        const volume = document.getElementById('proposal-volume').value || 100000;
        const cost = document.getElementById('proposal-cost').value || 50;

        const response = await fetch('/api/hoshin/projects/from-proposals/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({
                vsm_id: vsmId,
                proposals: selected,
                annual_volume: parseFloat(volume),
                cost_per_unit: parseFloat(cost)
            })
        });

        if (!response.ok) {
            const data = await response.json();
            alert(safeStr(data.error, 'Failed to create projects.'));
            return;
        }

        const data = await response.json();
        closeProposalModal();
        alert(`Created ${data.created} CI project(s). View them in Hoshin Kanri.`);
    } catch (err) {
        alert('Error creating projects: ' + safeStr(err, 'Unknown error'));
    }
}

async function approveProposal(burstId, title, savingsTarget, calcMethod, projectType) {
    try {
        const response = await fetch(`/api/vsm/${vsmId}/approve-proposal/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({
                burst_id: burstId,
                title: title,
                annual_savings_target: savingsTarget,
                calculation_method: calcMethod,
                project_type: projectType,
            }),
        });
        const data = await response.json();
        if (response.status === 409) {
            alert('This proposal has already been approved.');
            return;
        }
        if (!response.ok) {
            alert(safeStr(data.error, 'Failed to approve proposal.'));
            return;
        }
        closeProposalModal();
        alert(`Hoshin project created: ${data.title}\nSavings target: $${data.annual_savings_target.toLocaleString()}/yr\n\nView in Hoshin Kanri.`);
    } catch (err) {
        alert('Error: ' + safeStr(err, 'Unknown error'));
    }
}

// =============================================================================
// View Controls
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

function resetView() {
    zoom = 1;
    panX = 0;
    panY = 0;
    updateCanvasTransform();
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
function showNewVSMDialog() {
    document.getElementById('new-vsm-dialog').style.display = 'flex';
}

function hideNewVSMDialog() {
    document.getElementById('new-vsm-dialog').style.display = 'none';
}

function showVSMList(maps) {
    const container = document.getElementById('canvas-container');
    const emptyState = document.getElementById('empty-state');

    // Group by fiscal year
    const groups = {};
    maps.forEach(m => {
        const fy = m.fiscal_year || 'Unscoped';
        if (!groups[fy]) groups[fy] = [];
        groups[fy].push(m);
    });

    // Sort FY keys: numbered years descending, then "Unscoped" last
    const fyKeys = Object.keys(groups).sort((a, b) => {
        if (a === 'Unscoped') return 1;
        if (b === 'Unscoped') return -1;
        return parseInt(b) - parseInt(a);
    });

    let html = '<div class="vsm-list"><h3 style="margin-bottom:1rem;">Your Value Stream Maps</h3>';

    fyKeys.forEach(fy => {
        const fyLabel = fy === 'Unscoped' ? 'Unscoped' : `FY ${fy}`;
        html += `<div style="margin-bottom:1.5rem;">
            <h4 style="font-size:13px;color:var(--text-secondary);margin:0 0 8px;text-transform:uppercase;letter-spacing:0.5px;">${fyLabel}</h4>`;

        groups[fy].forEach(m => {
            const promoteBtn = m.status === 'future' && m.paired_with_id
                ? `<button class="sv-btn sv-btn-outline" style="font-size:10px;padding:2px 8px;margin-left:8px;" onclick="event.stopPropagation();promoteVSM('${m.id}')">Promote to Current</button>`
                : '';
            const pairedLabel = m.paired_with_id
                ? '<span style="font-size:10px;color:var(--text-dim);margin-left:6px;">paired</span>'
                : '';

            html += `
                <div class="vsm-item" onclick="window.location.href='/app/vsm/${m.id}/'">
                    <div class="vsm-title">${m.name}
                        <span class="vsm-status ${m.status}">${m.status}</span>${pairedLabel}${promoteBtn}
                    </div>
                    <div class="vsm-meta">${m.product_family || 'No product family'} &middot; Updated ${new Date(m.updated_at).toLocaleDateString()}</div>
                </div>
            `;
        });
        html += '</div>';
    });
    html += '</div>';

    emptyState.innerHTML = html + '<button class="sv-btn sv-btn-primary" style="margin:1rem;" onclick="showNewVSMDialog()">+ New VSM</button>';
}

async function promoteVSM(vsmId) {
    if (!confirm('Promote this future-state VSM to current? The old current state will be archived.')) return;
    try {
        const resp = await fetch(`/api/hoshin/vsm/${vsmId}/promote/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCookie('csrftoken') },
            body: JSON.stringify({ action: 'promote' }),
        });
        if (!resp.ok) {
            const err = await resp.json();
            alert(safeStr(err.error, 'Promotion failed'));
            return;
        }
        alert('VSM promoted to current state. Old current archived.');
        location.reload();
    } catch (e) {
        console.error('Promote VSM error:', e);
    }
}
