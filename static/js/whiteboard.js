// ============================================================================
// WHITEBOARD - Visual Mapping Tool
// whiteboard.js — init, state management, board CRUD
// ============================================================================

// --- Global State ---
let currentTool = 'select';
let currentColor = 'yellow';
let connectorStyle = 'orthogonal';
let selectedElement = null;
let selectedElements = new Set(); // Multi-select: stores element IDs
let elements = [];
let connections = [];
let zoom = 1;
let panX = 0, panY = 0;
let isPanning = false;
let isConnecting = false;
let connectionStart = null;
let elementIdCounter = 0;
let isDragging = false;
let didDrag = false;
let dragOffset = { x: 0, y: 0 };
let dragLastPos = { x: 0, y: 0 }; // For multi-drag delta tracking
let currentProjectId = null;
let isMarquee = false;
let marqueeStart = { x: 0, y: 0 };
let clipboard = { elements: [], connections: [] }; // Copy/paste buffer
let isResizing = false;
let resizeHandle = null;
let resizeStart = { x: 0, y: 0, width: 0, height: 0, elX: 0, elY: 0 };
let resizeElement = null;
const GRID_SIZE = 20;
function snapToGrid(val) { return Math.round(val / GRID_SIZE) * GRID_SIZE; }

// Undo/Redo history
let history = [];
let historyIndex = -1;
const MAX_HISTORY = 50;

// --- DOM References ---
const canvas = document.getElementById('canvas');
const canvasContainer = document.getElementById('canvas-container');
const connectionsSvg = document.getElementById('connections-svg');
const contextMenu = document.getElementById('context-menu');

// ============================================================================
// Undo/Redo System
// ============================================================================

function saveState() {
    // Clone current state
    const state = {
        elements: JSON.parse(JSON.stringify(elements)),
        connections: JSON.parse(JSON.stringify(connections)),
        elementIdCounter
    };

    // Remove any future states if we're in the middle of history
    if (historyIndex < history.length - 1) {
        history = history.slice(0, historyIndex + 1);
    }

    // Add new state
    history.push(state);

    // Limit history size
    if (history.length > MAX_HISTORY) {
        history.shift();
    } else {
        historyIndex++;
    }
}

function restoreState(state) {
    // Clear canvas
    canvas.querySelectorAll('.wb-element').forEach(el => el.remove());
    connectionsSvg.querySelectorAll('path').forEach(p => p.remove());

    // Restore data
    elements = [];
    connections = state.connections;
    elementIdCounter = state.elementIdCounter;
    selectedElement = null;

    // Recreate elements
    state.elements.forEach(el => {
        createElement(el.type, el.x, el.y, el, true); // true = skip saving state
    });

    // Render connections
    setTimeout(() => renderConnections(), 10);
}

function undo() {
    if (historyIndex > 0) {
        historyIndex--;
        restoreState(history[historyIndex]);
    }
}

function redo() {
    if (historyIndex < history.length - 1) {
        historyIndex++;
        restoreState(history[historyIndex]);
    }
}

// ============================================================================
// Tool Selection
// ============================================================================

document.querySelectorAll('.wb-tool-btn[data-tool]').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.wb-tool-btn[data-tool]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentTool = btn.dataset.tool;

        // Set cursor based on tool
        if (currentTool === 'pan') {
            canvasContainer.style.cursor = 'grab';
        } else if (currentTool === 'connect' || currentTool === 'causal') {
            canvasContainer.style.cursor = 'crosshair';
        } else {
            canvasContainer.style.cursor = 'default';
        }

        // Toggle causal mode class for styling
        document.querySelector('.wb-container').classList.toggle('causal-mode', currentTool === 'causal');

        // Toggle connect mode class for showing connection points
        const container = document.querySelector('.wb-container');
        if (currentTool === 'connect' || currentTool === 'causal') {
            container.classList.add('connect-mode');
        } else {
            container.classList.remove('connect-mode');
        }
    });
});

// Connector style selection
document.querySelectorAll('.wb-conn-style-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.wb-conn-style-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        connectorStyle = btn.dataset.connStyle;
    });
});

document.querySelectorAll('.wb-color-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.wb-color-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentColor = btn.dataset.color;
        console.log('Color changed to:', currentColor);

        // Update selected element color
        if (selectedElement && selectedElement.classList.contains('wb-postit')) {
            saveState();
            // Remove old color classes
            ['yellow', 'green', 'pink', 'orange', 'blue', 'purple'].forEach(c => {
                selectedElement.classList.remove(c);
            });
            // Add new color
            selectedElement.classList.add(currentColor);
            updateElementData(selectedElement.dataset.id, { color: currentColor });
        }
    });
});

// ============================================================================
// Context Menu
// ============================================================================

function showContextMenu(x, y) {
    contextMenu.style.left = `${x}px`;
    contextMenu.style.top = `${y}px`;
    contextMenu.classList.add('active');
}

function hideContextMenu() {
    contextMenu.classList.remove('active');
}

document.addEventListener('click', (e) => {
    if (!contextMenu.contains(e.target)) {
        hideContextMenu();
    }
});

function bringToFront() {
    if (!selectedElement) return;
    hideContextMenu();
    canvas.appendChild(selectedElement);
}

function sendToBack() {
    if (!selectedElement) return;
    hideContextMenu();
    canvas.insertBefore(selectedElement, canvas.firstChild);
}

function deleteSelected() {
    if (selectedElements.size === 0) return;
    hideContextMenu();
    saveState();

    const idsToDelete = new Set(selectedElements);

    // Remove connections to/from any deleted element
    connections = connections.filter(c =>
        !idsToDelete.has(c.from.elementId) && !idsToDelete.has(c.to.elementId)
    );

    // Remove element data and DOM elements
    idsToDelete.forEach(id => {
        const domEl = document.querySelector(`[data-id="${id}"]`);
        if (domEl) {
            removeResizeHandles(domEl);
            domEl.remove();
        }
    });
    elements = elements.filter(el => !idsToDelete.has(el.id));

    selectedElements.clear();
    selectedElement = null;
    renderConnections();
    document.getElementById('properties-panel').classList.remove('active');
    checkEmptyGuide();
}

// ============================================================================
// Copy / Paste
// ============================================================================

function copySelected() {
    if (selectedElements.size === 0) return;
    const ids = new Set(selectedElements);
    clipboard.elements = elements.filter(el => ids.has(el.id)).map(el => JSON.parse(JSON.stringify(el)));
    clipboard.connections = connections.filter(c =>
        ids.has(c.from.elementId) && ids.has(c.to.elementId)
    ).map(c => JSON.parse(JSON.stringify(c)));
}

function pasteClipboard() {
    if (clipboard.elements.length === 0) return;
    saveState();
    const idMap = {};
    const newEls = [];

    // Create elements with new IDs, offset by 40,40
    clipboard.elements.forEach(elData => {
        const oldId = elData.id;
        const newId = `el-${++elementIdCounter}`;
        idMap[oldId] = newId;
        const opts = { ...elData, id: newId };
        const newEl = createElement(elData.type, elData.x + 40, elData.y + 40, opts, true);
        newEls.push({ el: newEl, id: newId });
    });

    // Recreate connections with remapped IDs
    clipboard.connections.forEach(conn => {
        const fromId = idMap[conn.from.elementId];
        const toId = idMap[conn.to.elementId];
        if (fromId && toId) {
            createConnection(fromId, conn.from.position, toId, conn.to.position,
                { type: conn.type, style: conn.style }, true);
        }
    });

    // Select all pasted elements
    deselectAll();
    newEls.forEach(({ el, id }) => {
        selectedElements.add(id);
        el.classList.add('selected');
        selectedElement = el;
    });
    updateSelectionInfo();

    // Shift clipboard offset for subsequent pastes
    clipboard.elements.forEach(el => { el.x += 40; el.y += 40; });
    renderConnections();
}

function duplicateSelected() {
    if (selectedElements.size === 0) return;
    hideContextMenu();
    copySelected();
    pasteClipboard();
}

// ============================================================================
// Properties Panel
// ============================================================================

function showProperties(el) {
    const panel = document.getElementById('properties-panel');
    const content = document.getElementById('properties-content');
    const data = elements.find(e => e.id === el.dataset.id);

    if (!data) return;

    let html = '';

    if (data.type === 'postit' || data.type === 'text') {
        html = `
            <div class="sv-field">
                <div class="sv-label">Content</div>
                <textarea class="sv-input" style="height:80px;resize:vertical;"
                    onchange="updateElementContent('${data.id}', this.value)">${el.textContent}</textarea>
            </div>
        `;
    }

    if (data.type === 'group') {
        html = `
            <div class="sv-field">
                <div class="sv-label">Title</div>
                <input class="sv-input" type="text" value="${data.title || ''}"
                    onchange="updateGroupTitle('${data.id}', this.value)">
            </div>
            <div class="sv-field">
                <div class="sv-label">Width</div>
                <input class="sv-input" type="number" value="${data.width || 250}"
                    onchange="updateGroupSize('${data.id}', this.value, null)">
            </div>
            <div class="sv-field">
                <div class="sv-label">Height</div>
                <input class="sv-input" type="number" value="${data.height || 200}"
                    onchange="updateGroupSize('${data.id}', null, this.value)">
            </div>
        `;
    }

    html += `
        <div class="sv-field">
            <div class="sv-label">Position</div>
            <div style="display:flex;gap:8px;">
                <input class="sv-input" type="number" value="${Math.round(data.x)}" placeholder="X"
                    onchange="updateElementPosition('${data.id}', this.value, null)" style="flex:1;">
                <input class="sv-input" type="number" value="${Math.round(data.y)}" placeholder="Y"
                    onchange="updateElementPosition('${data.id}', null, this.value)" style="flex:1;">
            </div>
        </div>
    `;

    content.innerHTML = html;
    panel.classList.add('active');
}

function updateElementData(id, updates) {
    const idx = elements.findIndex(el => el.id === id);
    if (idx >= 0) {
        elements[idx] = { ...elements[idx], ...updates };
    }
}

function updateElementContent(id, text) {
    const el = document.querySelector(`[data-id="${id}"]`);
    if (el) {
        el.textContent = text;
        updateElementData(id, { text });
    }
}

function updateElementPosition(id, x, y) {
    const el = document.querySelector(`[data-id="${id}"]`);
    if (el) {
        if (x !== null) el.style.left = `${x}px`;
        if (y !== null) el.style.top = `${y}px`;
        updateElementData(id, {
            x: x !== null ? parseFloat(x) : elements.find(e => e.id === id)?.x,
            y: y !== null ? parseFloat(y) : elements.find(e => e.id === id)?.y
        });
        renderConnections();
    }
}

function updateGroupTitle(id, title) {
    const el = document.querySelector(`[data-id="${id}"]`);
    if (el) {
        el.querySelector('.wb-group-header').textContent = title;
        updateElementData(id, { title });
    }
}

function updateGroupSize(id, width, height) {
    const el = document.querySelector(`[data-id="${id}"]`);
    if (el) {
        if (width !== null) el.style.width = `${width}px`;
        if (height !== null) el.style.height = `${height}px`;
        updateElementData(id, {
            width: width !== null ? parseFloat(width) : elements.find(e => e.id === id)?.width,
            height: height !== null ? parseFloat(height) : elements.find(e => e.id === id)?.height
        });
    }
}

// ============================================================================
// Templates
// ============================================================================

function clearCanvas(resetHistory = false) {
    // Remove elements but preserve the SVG
    canvas.querySelectorAll('.wb-element').forEach(el => el.remove());
    connectionsSvg.querySelectorAll('path').forEach(p => p.remove());
    elements = [];
    connections = [];
    elementIdCounter = 0;
    selectedElement = null;

    if (resetHistory) {
        history = [];
        historyIndex = -1;
    }
    checkEmptyGuide();
}

function loadTemplate(template) {
    clearCanvas(true); // Reset history for templates

    switch (template) {
        case 'affinity':
            // Create groups
            createElement('group', 50, 50, { title: 'Theme 1', width: 280, height: 300 });
            createElement('group', 360, 50, { title: 'Theme 2', width: 280, height: 300 });
            createElement('group', 670, 50, { title: 'Theme 3', width: 280, height: 300 });

            // Add sample post-its
            createElement('postit', 70, 120, { color: 'yellow', text: 'Idea 1' });
            createElement('postit', 70, 240, { color: 'yellow', text: 'Idea 2' });
            createElement('postit', 380, 120, { color: 'green', text: 'Idea 3' });
            createElement('postit', 690, 120, { color: 'pink', text: 'Idea 4' });
            break;

        case 'interrelationship':
            // Create nodes in a circle
            const centerX = 400, centerY = 250, radius = 180;
            const nodes = ['Factor A', 'Factor B', 'Factor C', 'Factor D', 'Factor E'];
            nodes.forEach((name, i) => {
                const angle = (i / nodes.length) * 2 * Math.PI - Math.PI/2;
                const x = centerX + radius * Math.cos(angle) - 60;
                const y = centerY + radius * Math.sin(angle) - 30;
                createElement('rectangle', x, y, { text: name });
            });
            break;

        case 'process':
            createElement('oval', 50, 150, { text: 'Start' });
            createElement('rectangle', 200, 140, { text: 'Step 1' });
            createElement('diamond', 370, 130, { text: 'Decision?' });
            createElement('rectangle', 500, 140, { text: 'Step 2' });
            createElement('oval', 670, 150, { text: 'End' });

            // Add connections
            setTimeout(() => {
                createConnection('el-1', 'right', 'el-2', 'left');
                createConnection('el-2', 'right', 'el-3', 'left');
                createConnection('el-3', 'right', 'el-4', 'left');
                createConnection('el-4', 'right', 'el-5', 'left');
            }, 100);
            break;

        case 'fishbone':
            createFishbone(50, 100, 'Hypothesis / Effect');
            break;
    }

    // Save initial state after template load
    setTimeout(() => saveState(), 100);
}

// ============================================================================
// Drag and Drop from Sidebar
// ============================================================================

document.querySelectorAll('.wb-shape-item[draggable="true"]').forEach(item => {
    item.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('shape', item.dataset.shape);
    });
});

// ============================================================================
// Keyboard Shortcuts
// ============================================================================

document.addEventListener('keydown', (e) => {
    // Don't handle if typing in an editable element
    if (e.target.contentEditable === 'true' || e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
    }

    const mod = e.ctrlKey || e.metaKey;
    switch (e.key.toLowerCase()) {
        case 'c':
            if (mod) { e.preventDefault(); copySelected(); }
            else { document.querySelector('[data-tool="connect"]').click(); }
            break;
        case 'v':
            if (mod) { e.preventDefault(); pasteClipboard(); }
            else { document.querySelector('[data-tool="select"]').click(); }
            break;
        case 'd':
            if (mod) { e.preventDefault(); duplicateSelected(); }
            break;
        case 'a':
            if (mod) {
                e.preventDefault();
                // Select all elements
                elements.forEach(elData => {
                    const domEl = document.querySelector(`[data-id="${elData.id}"]`);
                    if (domEl) {
                        selectedElements.add(elData.id);
                        domEl.classList.add('selected');
                        selectedElement = domEl;
                    }
                });
                updateSelectionInfo();
            }
            break;
        case 'p':
            if (!mod) document.querySelector('[data-tool="postit"]').click();
            break;
        case 'i':
            if (!mod) document.querySelector('[data-tool="causal"]').click();
            break;
        case 't':
            if (!mod) document.querySelector('[data-tool="text"]').click();
            break;
        case 'g':
            if (!mod) document.querySelector('[data-tool="group"]').click();
            break;
        case 'delete':
        case 'backspace':
            if (selectedElements.size > 0) {
                deleteSelected();
            }
            break;
        case '=':
        case '+':
            zoomIn();
            break;
        case '-':
            zoomOut();
            break;
        case '0':
            zoomReset();
            break;
        case 's':
            if (mod) { e.preventDefault(); saveBoard(); }
            break;
        case 'z':
            if (mod) {
                e.preventDefault();
                if (e.shiftKey) { redo(); } else { undo(); }
            }
            break;
        case 'y':
            if (mod) { e.preventDefault(); redo(); }
            break;
        case 'f5':
            e.preventDefault();
            togglePresentation();
            break;
        case 'escape':
            if (isConnecting) {
                cancelConnection();
            } else {
                deselectAll();
                currentTool = 'select';
                document.querySelector('[data-tool="select"]').click();
            }
            break;
    }

    // Space for panning
    if (e.code === 'Space' && !isPanning) {
        e.preventDefault();
        canvasContainer.style.cursor = 'grab';
    }
});

document.addEventListener('keyup', (e) => {
    if (e.code === 'Space') {
        canvasContainer.style.cursor = currentTool === 'pan' ? 'grab' : 'default';
    }
});

// ============================================================================
// Empty Board Guidance
// ============================================================================

function checkEmptyGuide() {
    const guide = document.getElementById('empty-guide');
    if (!guide) return;
    guide.style.display = elements.length === 0 ? '' : 'none';
}

// ============================================================================
// Presentation Mode
// ============================================================================

let isPresentationMode = false;

function togglePresentation() {
    const container = document.querySelector('.wb-container');
    if (!isPresentationMode) {
        isPresentationMode = true;
        container.classList.add('wb-presentation');
        if (container.requestFullscreen) container.requestFullscreen();
    } else {
        isPresentationMode = false;
        container.classList.remove('wb-presentation');
        if (document.fullscreenElement) document.exitFullscreen();
    }
    const btn = document.getElementById('present-btn');
    if (btn) btn.classList.toggle('active', isPresentationMode);
}

// Exit presentation mode when user exits fullscreen via browser UI / Escape
document.addEventListener('fullscreenchange', () => {
    if (!document.fullscreenElement && isPresentationMode) {
        isPresentationMode = false;
        document.querySelector('.wb-container').classList.remove('wb-presentation');
        const btn = document.getElementById('present-btn');
        if (btn) btn.classList.remove('active');
    }
});

// ============================================================================
// Save/Load (Local)
// ============================================================================

async function saveBoard() {
    const boardData = {
        elements,
        connections,
        zoom,
        panX,
        panY,
        timestamp: new Date().toISOString()
    };

    const name = await svendPrompt('Board name:', 'Untitled Board');
    if (!name) return;

    // Save to localStorage for now
    const boards = JSON.parse(localStorage.getItem('wb-boards') || '{}');
    boards[name] = boardData;
    localStorage.setItem('wb-boards', JSON.stringify(boards));

    loadBoardsList();
    alert('Board saved!');
}

function loadBoard(name) {
    const boards = JSON.parse(localStorage.getItem('wb-boards') || '{}');
    const data = boards[name];
    if (!data) return;

    clearCanvas(true); // Reset history when loading board
    connections = data.connections || [];
    zoom = data.zoom || 1;
    panX = data.panX || 0;
    panY = data.panY || 0;
    elementIdCounter = 0;

    updateCanvasTransform();
    document.getElementById('zoom-display').textContent = `${Math.round(zoom * 100)}%`;

    // Recreate elements
    data.elements.forEach(el => {
        createElement(el.type, el.x, el.y, el);
        elementIdCounter = Math.max(elementIdCounter, parseInt(el.id.split('-')[1]) || 0);
    });

    // Recreate connections
    setTimeout(() => renderConnections(), 100);
}

function loadBoardsList() {
    const boards = JSON.parse(localStorage.getItem('wb-boards') || '{}');
    const list = document.getElementById('boards-list');

    if (Object.keys(boards).length === 0) {
        list.innerHTML = '<div style="color:var(--wb-text-dim);font-size:11px;text-align:center;padding:20px;">No saved boards yet</div>';
        return;
    }

    list.innerHTML = Object.keys(boards).map(name => `
        <div class="wb-template-item" style="position:relative;">
            <div class="icon" onclick="loadBoard('${name}')" style="cursor:pointer;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="3" width="18" height="18" rx="2"/>
                </svg>
            </div>
            <div class="info" onclick="loadBoard('${name}')" style="cursor:pointer;">
                <div class="name">${name}</div>
                <div class="desc">${new Date(boards[name].timestamp).toLocaleDateString()}</div>
            </div>
            <button onclick="deleteBoard('${name}')" title="Delete board" style="
                position:absolute;right:8px;top:50%;transform:translateY(-50%);
                background:transparent;border:none;color:#9aaa9a;cursor:pointer;
                padding:4px;opacity:0.5;transition:opacity 0.15s;
            " onmouseover="this.style.opacity='1';this.style.color='#e85747'" onmouseout="this.style.opacity='0.5';this.style.color='#9aaa9a'">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
            </button>
        </div>
    `).join('');
}

function deleteBoard(name) {
    if (!confirm(`Delete board "${name}"?`)) return;

    const boards = JSON.parse(localStorage.getItem('wb-boards') || '{}');
    delete boards[name];
    localStorage.setItem('wb-boards', JSON.stringify(boards));
    loadBoardsList();
}

function newBoard() {
    if (elements.length > 0 && !confirm('Clear current board?')) return;

    clearCanvas();
    zoom = 1;
    panX = 0;
    panY = 0;
    updateCanvasTransform();
    document.getElementById('zoom-display').textContent = '100%';
}

// ============================================================================
// Affinity to Interrelationship Conversion
// ============================================================================

function convertAffinityToIR() {
    // Find all group elements and extract their titles
    const groups = elements.filter(el => el.type === 'group');

    if (groups.length === 0) {
        alert('No groups found. Create an Affinity Diagram with groups first.');
        return;
    }

    if (groups.length < 2) {
        alert('Need at least 2 groups to create an Interrelationship Diagram.');
        return;
    }

    // Extract titles from groups
    const factors = groups.map(g => {
        const domEl = document.querySelector(`[data-id="${g.id}"]`);
        const header = domEl?.querySelector('.wb-group-header');
        return header?.textContent || g.title || 'Factor';
    });

    if (!confirm(`Convert ${factors.length} groups to Interrelationship Diagram?\n\nFactors: ${factors.join(', ')}`)) {
        return;
    }

    // Clear canvas
    clearCanvas();

    // Calculate layout - arrange in a circle
    const centerX = 450;
    const centerY = 300;
    const radius = Math.min(250, 80 + factors.length * 30);

    factors.forEach((name, i) => {
        const angle = (i / factors.length) * 2 * Math.PI - Math.PI/2;
        const x = centerX + radius * Math.cos(angle) - 60;
        const y = centerY + radius * Math.sin(angle) - 30;
        createElement('rectangle', x, y, { text: name });
    });

    // Add instruction text
    createElement('text', centerX - 100, centerY - 10, {
        text: 'Draw arrows to show cause → effect'
    });

    // Switch to connect tool
    document.querySelector('[data-tool="connect"]').click();
}

function analyzeIR() {
    // Count incoming and outgoing connections for each element
    const shapes = elements.filter(el =>
        ['rectangle', 'diamond', 'oval', 'parallelogram', 'document', 'cylinder'].includes(el.type)
    );

    if (shapes.length === 0) {
        alert('No shapes found to analyze. Create an Interrelationship Diagram first.');
        return;
    }

    if (connections.length === 0) {
        alert('No connections found. Draw arrows between factors to show cause-effect relationships.');
        return;
    }

    // Count arrows for each element
    const analysis = shapes.map(shape => {
        const outgoing = connections.filter(c => c.from.elementId === shape.id).length;
        const incoming = connections.filter(c => c.to.elementId === shape.id).length;
        const domEl = document.querySelector(`[data-id="${shape.id}"]`);
        const text = domEl?.querySelector('span')?.textContent || shape.text || 'Unknown';

        return {
            id: shape.id,
            text,
            outgoing,
            incoming,
            net: outgoing - incoming
        };
    });

    // Sort by net (drivers first, outcomes last)
    analysis.sort((a, b) => b.net - a.net);

    // Build results message
    let msg = 'INTERRELATIONSHIP ANALYSIS\n\n';

    analysis.forEach(item => {
        const net = (item.net >= 0 ? '+' : '') + item.net;
        msg += `${item.text}: Out=${item.outgoing}  In=${item.incoming}  Net=${net}\n`;
    });

    // Identify drivers and outcomes
    const drivers = analysis.filter(a => a.net > 0);
    const outcomes = analysis.filter(a => a.net < 0);
    const neutral = analysis.filter(a => a.net === 0);

    msg += '\n';
    if (drivers.length > 0) {
        msg += 'DRIVERS (more outgoing):\n';
        drivers.forEach(d => msg += `  - ${d.text} (net ${d.net > 0 ? '+' : ''}${d.net})\n`);
    }

    if (outcomes.length > 0) {
        msg += 'OUTCOMES (more incoming):\n';
        outcomes.forEach(d => msg += `  - ${d.text} (net ${d.net})\n`);
    }

    if (neutral.length > 0) {
        msg += 'NEUTRAL:\n';
        neutral.forEach(d => msg += `  - ${d.text}\n`);
    }

    msg += '\nFocus improvement efforts on DRIVERS for maximum downstream impact.';

    alert(msg);

    // Also add visual indicators to the diagram
    analysis.forEach(item => {
        const domEl = document.querySelector(`[data-id="${item.id}"]`);
        if (!domEl) return;

        // Remove existing badges
        domEl.querySelectorAll('.ir-badge').forEach(b => b.remove());

        // Add badge
        const badge = document.createElement('div');
        badge.className = 'ir-badge';
        badge.style.cssText = `
            position: absolute;
            top: -12px;
            right: -12px;
            min-width: 24px;
            height: 24px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0 6px;
            z-index: 10;
        `;

        if (item.net > 0) {
            badge.style.background = '#e85747';
            badge.style.color = 'white';
            badge.textContent = `+${item.net}`;
            badge.title = 'Driver: ' + item.outgoing + ' out, ' + item.incoming + ' in';
        } else if (item.net < 0) {
            badge.style.background = '#47a5e8';
            badge.style.color = 'white';
            badge.textContent = item.net;
            badge.title = 'Outcome: ' + item.outgoing + ' out, ' + item.incoming + ' in';
        } else {
            badge.style.background = '#9aaa9a';
            badge.style.color = '#0a0f0a';
            badge.textContent = '0';
            badge.title = 'Neutral: ' + item.outgoing + ' out, ' + item.incoming + ' in';
        }

        domEl.appendChild(badge);
    });
}
