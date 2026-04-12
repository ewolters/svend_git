// ============================================================================
// wb-canvas.js — element creation, selection, drag, resize, zoom/pan
// ============================================================================

// ============================================================================
// Element Creation
// ============================================================================

function createElement(type, x, y, options = {}, skipHistory = false) {
    if (!skipHistory) saveState();
    const id = options.id || `el-${++elementIdCounter}`;
    const el = document.createElement('div');
    el.className = 'wb-element';
    el.dataset.id = id;
    el.dataset.type = type;
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;

    const elementData = {
        id,
        type,
        x,
        y,
        ...options
    };

    switch (type) {
        case 'postit':
            const postitColor = options.color || currentColor;
            el.classList.add('wb-postit', postitColor);
            el.contentEditable = 'true';
            el.textContent = options.text || 'Double-click to edit';
            elementData.color = postitColor;
            elementData.text = options.text || '';
            console.log('Created postit with color:', postitColor, 'classes:', el.className);
            break;

        case 'text':
            el.className += ' wb-text';
            el.contentEditable = 'true';
            el.textContent = options.text || 'Text';
            elementData.text = options.text || 'Text';
            break;

        case 'group':
            el.className += ' wb-group';
            el.innerHTML = `<div class="wb-group-header" contenteditable="true">${options.title || 'Group Title'}</div>`;
            el.style.width = `${options.width || 250}px`;
            el.style.height = `${options.height || 200}px`;
            elementData.title = options.title || 'Group Title';
            elementData.width = options.width || 250;
            elementData.height = options.height || 200;
            break;

        case 'rectangle':
        case 'diamond':
        case 'oval':
        case 'parallelogram':
            el.className += ` wb-process-shape ${type}`;
            el.innerHTML = `<span contenteditable="true">${options.text || 'Step'}</span>`;
            elementData.text = options.text || 'Step';
            break;

        case 'document':
            el.className += ' wb-process-shape document';
            el.innerHTML = `<span contenteditable="true">${options.text || 'Document'}</span>`;
            elementData.text = options.text || 'Document';
            break;

        case 'cylinder':
            el.className += ' wb-process-shape cylinder';
            el.innerHTML = `<span contenteditable="true" style="position:relative;z-index:3;">${options.text || 'Database'}</span>`;
            elementData.text = options.text || 'Database';
            break;

        case 'gate-and':
            el.className += ' wb-gate and-gate';
            el.dataset.gateType = 'AND';
            elementData.gateType = 'AND';
            break;

        case 'gate-or':
            el.className += ' wb-gate or-gate';
            el.dataset.gateType = 'OR';
            elementData.gateType = 'OR';
            break;

        case 'image':
            el.className += ' wb-image';
            const img = document.createElement('img');
            img.src = options.src || '';
            img.draggable = false;
            el.appendChild(img);
            el.style.width = `${options.width || 200}px`;
            el.style.height = `${options.height || 150}px`;
            elementData.src = options.src || '';
            elementData.width = options.width || 200;
            elementData.height = options.height || 150;
            elementData.originalWidth = options.originalWidth || elementData.width;
            elementData.originalHeight = options.originalHeight || elementData.height;
            break;

        case 'fishbone':
            // Fishbone is created separately, but handle restore
            el.className += ' wb-fishbone';
            elementData.effect = options.effect || 'Hypothesis / Effect';
            elementData.categories = options.categories || [];
            renderFishbone(el, elementData);
            // Skip adding connection points for fishbone
            canvas.appendChild(el);
            elements.push(elementData);
            return el;
    }

    // Apply stored width/height if present (for resize restore)
    if (options.width && type !== 'image' && type !== 'group') {
        el.style.width = `${options.width}px`;
        elementData.width = options.width;
    }
    if (options.height && type !== 'image' && type !== 'group') {
        el.style.height = `${options.height}px`;
        elementData.height = options.height;
        if (type === 'postit') el.style.minHeight = `${options.height}px`;
    }

    // Add connection points
    addConnectionPoints(el);

    // Event listeners
    el.addEventListener('mousedown', (e) => onElementMouseDown(e, el));
    el.addEventListener('dblclick', (e) => onElementDoubleClick(e, el));
    el.addEventListener('contextmenu', (e) => onElementContextMenu(e, el));

    canvas.appendChild(el);
    elements.push(elementData);

    // Hide empty board guidance
    const guide = document.getElementById('empty-guide');
    if (guide) guide.style.display = 'none';

    return el;
}

function addConnectionPoints(el) {
    ['top', 'bottom', 'left', 'right'].forEach(pos => {
        const point = document.createElement('div');
        point.className = `wb-connection-point ${pos}`;
        point.addEventListener('mousedown', (e) => {
            e.stopPropagation();
            startConnection(e, el, pos);
        });
        point.addEventListener('mouseup', (e) => {
            e.stopPropagation();
            if (isConnecting && connectionStart) {
                finishConnection(el, pos);
            }
        });
        point.addEventListener('mouseenter', (e) => {
            if (isConnecting) {
                point.style.transform = pos === 'top' || pos === 'bottom'
                    ? 'translateX(-50%) scale(1.5)'
                    : 'translateY(-50%) scale(1.5)';
                point.style.background = '#5fc484';
            }
        });
        point.addEventListener('mouseleave', (e) => {
            point.style.transform = '';
            point.style.background = '';
        });
        el.appendChild(point);
    });
}

// ============================================================================
// Element Interaction
// ============================================================================

function onElementMouseDown(e, el) {
    if (e.target.classList.contains('wb-connection-point')) return;
    if (e.target.classList.contains('wb-resize-handle')) return;
    if (el.contentEditable === 'true' && document.activeElement === el) return;

    e.stopPropagation();

    if (e.shiftKey) {
        selectElement(el, true); // Additive selection
    } else if (!selectedElements.has(el.dataset.id)) {
        selectElement(el); // Normal single select (only if not already selected)
    } else {
        // Already selected — just set as primary for properties
        selectedElement = el;
    }

    if (currentTool === 'select') {
        isDragging = true;
        didDrag = false;
        const canvasRect = canvasContainer.getBoundingClientRect();
        // Store initial canvas position for delta-based multi-drag
        dragLastPos.x = (e.clientX - canvasRect.left - panX) / zoom;
        dragLastPos.y = (e.clientY - canvasRect.top - panY) / zoom;
        // Also store offset for single-element positioning
        const rect = el.getBoundingClientRect();
        dragOffset.x = e.clientX - rect.left;
        dragOffset.y = e.clientY - rect.top;
        el.style.cursor = 'grabbing';
    }
}

function onElementDoubleClick(e, el) {
    if (el.contentEditable === 'true') {
        el.focus();
        // Select all text
        const range = document.createRange();
        range.selectNodeContents(el);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
    }
}

function onElementContextMenu(e, el) {
    e.preventDefault();
    selectElement(el);
    showContextMenu(e.clientX, e.clientY);
}

function selectElement(el, addToSelection = false) {
    if (addToSelection) {
        // Shift+click: toggle in/out of multi-selection
        const id = el.dataset.id;
        if (selectedElements.has(id)) {
            selectedElements.delete(id);
            el.classList.remove('selected');
            removeResizeHandles(el);
            if (selectedElement === el) {
                selectedElement = null;
                document.getElementById('properties-panel').classList.remove('active');
            }
        } else {
            selectedElements.add(id);
            el.classList.add('selected');
            selectedElement = el;
            showProperties(el);
        }
        updateSelectionInfo();
        return;
    }
    // Normal click: clear multi-selection, select only this
    deselectAll();
    selectedElement = el;
    selectedElements.add(el.dataset.id);
    el.classList.add('selected');
    showProperties(el);
    addResizeHandles(el);
}

function deselectAll() {
    selectedElements.forEach(id => {
        const el = document.querySelector(`[data-id="${id}"]`);
        if (el) {
            el.classList.remove('selected');
            removeResizeHandles(el);
        }
    });
    selectedElements.clear();
    selectedElement = null;
    hideContextMenu();
    document.getElementById('properties-panel').classList.remove('active');
}

function updateSelectionInfo() {
    // Remove resize handles when multi-selecting (resize is single-element only)
    if (selectedElements.size > 1) {
        selectedElements.forEach(id => {
            const el = document.querySelector(`[data-id="${id}"]`);
            if (el) removeResizeHandles(el);
        });
    } else if (selectedElements.size === 1 && selectedElement) {
        addResizeHandles(selectedElement);
    }
}

// ============================================================================
// Canvas Events
// ============================================================================

canvasContainer.addEventListener('mousedown', (e) => {
    if (e.target === canvasContainer || e.target === canvas || e.target.id === 'selection-marquee') {
        if (!e.shiftKey) deselectAll();

        if (currentTool === 'pan' || e.button === 1) {
            isPanning = true;
            canvasContainer.style.cursor = 'grabbing';
        } else if (currentTool === 'select' && e.button === 0) {
            // Start marquee selection
            isMarquee = true;
            const rect = canvasContainer.getBoundingClientRect();
            marqueeStart.x = e.clientX - rect.left;
            marqueeStart.y = e.clientY - rect.top;
            const marquee = document.getElementById('selection-marquee');
            marquee.style.left = `${marqueeStart.x}px`;
            marquee.style.top = `${marqueeStart.y}px`;
            marquee.style.width = '0px';
            marquee.style.height = '0px';
        } else if (currentTool === 'postit') {
            const rect = canvasContainer.getBoundingClientRect();
            const x = snapToGrid((e.clientX - rect.left - panX) / zoom);
            const y = snapToGrid((e.clientY - rect.top - panY) / zoom);
            const el = createElement('postit', x, y);
            selectElement(el);
        } else if (currentTool === 'text') {
            const rect = canvasContainer.getBoundingClientRect();
            const x = snapToGrid((e.clientX - rect.left - panX) / zoom);
            const y = snapToGrid((e.clientY - rect.top - panY) / zoom);
            const el = createElement('text', x, y);
            selectElement(el);
        } else if (currentTool === 'group') {
            const rect = canvasContainer.getBoundingClientRect();
            const x = (e.clientX - rect.left - panX) / zoom;
            const y = (e.clientY - rect.top - panY) / zoom;
            const el = createElement('group', x, y);
            selectElement(el);
        }
    }
});

canvasContainer.addEventListener('mousemove', (e) => {
    if (isPanning) {
        panX += e.movementX;
        panY += e.movementY;
        updateCanvasTransform();
    } else if (isResizing && resizeElement) {
        handleResize(e);
    } else if (isDragging && selectedElement) {
        if (!didDrag) {
            saveState();
            didDrag = true;
        }
        const rect = canvasContainer.getBoundingClientRect();
        const curX = (e.clientX - rect.left - panX) / zoom;
        const curY = (e.clientY - rect.top - panY) / zoom;

        if (selectedElements.size > 1) {
            // Multi-drag: move all selected by delta
            const dx = snapToGrid(curX - dragLastPos.x);
            const dy = snapToGrid(curY - dragLastPos.y);
            if (dx !== 0 || dy !== 0) {
                selectedElements.forEach(id => {
                    const domEl = document.querySelector(`[data-id="${id}"]`);
                    const data = elements.find(el => el.id === id);
                    if (domEl && data) {
                        const newX = data.x + dx;
                        const newY = data.y + dy;
                        domEl.style.left = `${newX}px`;
                        domEl.style.top = `${newY}px`;
                        updateElementData(id, { x: newX, y: newY });
                    }
                });
                dragLastPos.x += dx;
                dragLastPos.y += dy;
                renderConnections();
            }
        } else {
            // Single drag: absolute positioning
            const rawX = curX - dragOffset.x;
            const rawY = curY - dragOffset.y;
            const x = snapToGrid(rawX);
            const y = snapToGrid(rawY);
            selectedElement.style.left = `${x}px`;
            selectedElement.style.top = `${y}px`;
            updateElementData(selectedElement.dataset.id, { x, y });
            renderConnections();
        }
    } else if (isMarquee) {
        const rect = canvasContainer.getBoundingClientRect();
        const curX = e.clientX - rect.left;
        const curY = e.clientY - rect.top;
        const marquee = document.getElementById('selection-marquee');
        const left = Math.min(marqueeStart.x, curX);
        const top = Math.min(marqueeStart.y, curY);
        const width = Math.abs(curX - marqueeStart.x);
        const height = Math.abs(curY - marqueeStart.y);
        if (width > 5 || height > 5) {
            marquee.style.display = 'block';
            marquee.style.left = `${left}px`;
            marquee.style.top = `${top}px`;
            marquee.style.width = `${width}px`;
            marquee.style.height = `${height}px`;
        }
    } else if (isConnecting) {
        updateTempConnection(e);
    }
});

canvasContainer.addEventListener('mouseup', (e) => {
    if (isPanning) {
        isPanning = false;
        canvasContainer.style.cursor = currentTool === 'pan' ? 'grab' : 'default';
    }
    if (isResizing) {
        isResizing = false;
        resizeElement = null;
        resizeHandle = null;
    }
    if (isDragging) {
        isDragging = false;
        if (selectedElement) {
            selectedElement.style.cursor = 'move';
        }
    }
    if (isMarquee) {
        isMarquee = false;
        const marquee = document.getElementById('selection-marquee');
        const mRect = {
            left: parseFloat(marquee.style.left),
            top: parseFloat(marquee.style.top),
            width: parseFloat(marquee.style.width),
            height: parseFloat(marquee.style.height),
        };
        marquee.style.display = 'none';
        // Only select if marquee was actually dragged (not a simple click)
        if (mRect.width > 5 || mRect.height > 5) {
            const containerRect = canvasContainer.getBoundingClientRect();
            // Convert marquee screen rect to canvas coords
            const mCanvasLeft = (mRect.left - panX) / zoom;
            const mCanvasTop = (mRect.top - panY) / zoom;
            const mCanvasRight = (mRect.left + mRect.width - panX) / zoom;
            const mCanvasBottom = (mRect.top + mRect.height - panY) / zoom;
            elements.forEach(elData => {
                const domEl = document.querySelector(`[data-id="${elData.id}"]`);
                if (!domEl) return;
                const elRight = elData.x + domEl.offsetWidth;
                const elBottom = elData.y + domEl.offsetHeight;
                // Check intersection
                if (elData.x < mCanvasRight && elRight > mCanvasLeft &&
                    elData.y < mCanvasBottom && elBottom > mCanvasTop) {
                    selectedElements.add(elData.id);
                    domEl.classList.add('selected');
                    selectedElement = domEl; // Last one becomes primary
                }
            });
            updateSelectionInfo();
            if (selectedElements.size === 1 && selectedElement) {
                showProperties(selectedElement);
                addResizeHandles(selectedElement);
            }
        }
    }
    if (isConnecting) {
        const target = e.target;
        if (target.classList.contains('wb-connection-point')) {
            const el = target.closest('.wb-element');
            const pos = ['top', 'bottom', 'left', 'right'].find(p => target.classList.contains(p));
            finishConnection(el, pos);
        } else {
            cancelConnection();
        }
    }
});

canvasContainer.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    zoom = Math.max(0.25, Math.min(3, zoom + delta));
    updateCanvasTransform();
    document.getElementById('zoom-display').textContent = `${Math.round(zoom * 100)}%`;
});

// ============================================================================
// Zoom Controls
// ============================================================================

function zoomIn() {
    zoom = Math.min(3, zoom + 0.25);
    updateCanvasTransform();
    document.getElementById('zoom-display').textContent = `${Math.round(zoom * 100)}%`;
}

function zoomOut() {
    zoom = Math.max(0.25, zoom - 0.25);
    updateCanvasTransform();
    document.getElementById('zoom-display').textContent = `${Math.round(zoom * 100)}%`;
}

function zoomReset() {
    zoom = 1;
    panX = 0;
    panY = 0;
    updateCanvasTransform();
    document.getElementById('zoom-display').textContent = '100%';
}

function updateCanvasTransform() {
    canvas.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
    // SVG is now inside canvas, so it transforms automatically
}

// ============================================================================
// Resize Handles
// ============================================================================

const RESIZE_SKIP_TYPES = ['fishbone', 'gate-and', 'gate-or'];
const MIN_SIZES = {
    postit: { w: 80, h: 60 },
    text: { w: 60, h: 24 },
    group: { w: 120, h: 80 },
    image: { w: 40, h: 40 },
    _default: { w: 60, h: 40 },
};

function addResizeHandles(el) {
    if (!el) return;
    const type = el.dataset.type;
    if (RESIZE_SKIP_TYPES.includes(type)) return;
    // Don't add if already present
    if (el.querySelector('.wb-resize-handle')) return;
    ['nw', 'ne', 'sw', 'se'].forEach(corner => {
        const handle = document.createElement('div');
        handle.className = `wb-resize-handle ${corner}`;
        handle.addEventListener('mousedown', (e) => {
            e.stopPropagation();
            e.preventDefault();
            isResizing = true;
            resizeHandle = corner;
            resizeElement = el;
            const w = el.offsetWidth;
            const h = el.offsetHeight;
            const elData = elements.find(d => d.id === el.dataset.id);
            resizeStart = {
                x: e.clientX, y: e.clientY,
                width: w, height: h,
                elX: elData ? elData.x : parseFloat(el.style.left),
                elY: elData ? elData.y : parseFloat(el.style.top),
            };
            saveState();
        });
        el.appendChild(handle);
    });
}

function removeResizeHandles(el) {
    if (!el) return;
    el.querySelectorAll('.wb-resize-handle').forEach(h => h.remove());
}

function handleResize(e) {
    if (!resizeElement || !resizeHandle) return;
    const dx = (e.clientX - resizeStart.x) / zoom;
    const dy = (e.clientY - resizeStart.y) / zoom;
    const type = resizeElement.dataset.type;
    const min = MIN_SIZES[type] || MIN_SIZES._default;

    let newW = resizeStart.width;
    let newH = resizeStart.height;
    let newX = resizeStart.elX;
    let newY = resizeStart.elY;

    // Diamond: keep square (width = height)
    const isDiamond = type === 'diamond';
    // Image: lock aspect ratio
    const isImage = type === 'image';
    const aspect = resizeStart.width / resizeStart.height;

    if (resizeHandle.includes('e')) newW = resizeStart.width + dx;
    if (resizeHandle.includes('w')) { newW = resizeStart.width - dx; newX = resizeStart.elX + dx; }
    if (resizeHandle.includes('s')) newH = resizeStart.height + dy;
    if (resizeHandle.includes('n')) { newH = resizeStart.height - dy; newY = resizeStart.elY + dy; }

    // Enforce minimums
    if (newW < min.w) { if (resizeHandle.includes('w')) newX -= (min.w - newW); newW = min.w; }
    if (newH < min.h) { if (resizeHandle.includes('n')) newY -= (min.h - newH); newH = min.h; }

    // Constrain diamond to square
    if (isDiamond) { const s = Math.max(newW, newH); newW = s; newH = s; }

    // Lock aspect ratio for images
    if (isImage) {
        if (Math.abs(dx) > Math.abs(dy)) {
            newH = newW / aspect;
        } else {
            newW = newH * aspect;
        }
    }

    newW = snapToGrid(newW);
    newH = snapToGrid(newH);
    newX = snapToGrid(newX);
    newY = snapToGrid(newY);

    resizeElement.style.width = `${newW}px`;
    resizeElement.style.height = `${newH}px`;
    if (type === 'postit') {
        resizeElement.style.minHeight = `${newH}px`;
    }
    resizeElement.style.left = `${newX}px`;
    resizeElement.style.top = `${newY}px`;

    updateElementData(resizeElement.dataset.id, { x: newX, y: newY, width: newW, height: newH });
    renderConnections();
}

// ============================================================================
// Image Drop / Paste
// ============================================================================

function processImageFile(file, canvasX, canvasY) {
    if (!file.type.startsWith('image/')) return;
    if (file.size > 5 * 1024 * 1024) {
        alert('Image too large. Maximum size is 5MB.');
        return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
        const img = new Image();
        img.onload = () => {
            // Downscale if needed (max 800px on either axis)
            let w = img.width;
            let h = img.height;
            const MAX_DIM = 800;
            if (w > MAX_DIM || h > MAX_DIM) {
                const scale = MAX_DIM / Math.max(w, h);
                w = Math.round(w * scale);
                h = Math.round(h * scale);
            }
            // Compress via offscreen canvas
            const offscreen = document.createElement('canvas');
            offscreen.width = w;
            offscreen.height = h;
            const octx = offscreen.getContext('2d');
            octx.drawImage(img, 0, 0, w, h);
            const dataUrl = offscreen.toDataURL('image/jpeg', 0.85);

            // Check final size (base64 in JSON)
            if (dataUrl.length > 2 * 1024 * 1024) {
                alert('Image too large after processing. Try a smaller image.');
                return;
            }

            const x = snapToGrid(canvasX - w / 2);
            const y = snapToGrid(canvasY - h / 2);
            const el = createElement('image', x, y, { src: dataUrl, width: w, height: h, originalWidth: w, originalHeight: h });
            selectElement(el);
        };
        img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
}

canvasContainer.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
});

canvasContainer.addEventListener('drop', (e) => {
    e.preventDefault();

    // Handle shape drag from sidebar
    const shape = e.dataTransfer.getData('shape');
    if (shape) {
        const rect = canvasContainer.getBoundingClientRect();
        const x = snapToGrid((e.clientX - rect.left - panX) / zoom);
        const y = snapToGrid((e.clientY - rect.top - panY) / zoom);
        const el = createElement(shape, x, y);
        selectElement(el);
        return;
    }

    // Handle image drop
    const files = e.dataTransfer.files;
    if (files.length === 0) return;
    const file = files[0];
    if (!file.type.startsWith('image/')) return;
    const rect = canvasContainer.getBoundingClientRect();
    const canvasX = (e.clientX - rect.left - panX) / zoom;
    const canvasY = (e.clientY - rect.top - panY) / zoom;
    processImageFile(file, canvasX, canvasY);
});

document.addEventListener('paste', (e) => {
    // Skip if typing in an editable field
    if (e.target.contentEditable === 'true' || e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith('image/')) {
            e.preventDefault();
            const file = items[i].getAsFile();
            // Paste at center of current viewport
            const rect = canvasContainer.getBoundingClientRect();
            const canvasX = (rect.width / 2 - panX) / zoom;
            const canvasY = (rect.height / 2 - panY) / zoom;
            processImageFile(file, canvasX, canvasY);
            break;
        }
    }
});
