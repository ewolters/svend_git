// ============================================================================
// wb-connections.js — connection routing (straight, orthogonal, curved)
// ============================================================================

let tempConnectionLine = null;

function startConnection(e, el, position) {
    e.stopPropagation();
    e.preventDefault();
    isConnecting = true;
    connectionStart = { element: el, position };
    canvasContainer.style.cursor = 'crosshair';

    // Create temporary connection line
    tempConnectionLine = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    tempConnectionLine.classList.add('wb-temp-connection');
    const startPoint = getConnectionPoint(el, position);
    connectionStart.startPoint = startPoint;
    tempConnectionLine.setAttribute('d', `M ${startPoint.x} ${startPoint.y} L ${startPoint.x} ${startPoint.y}`);
    connectionsSvg.appendChild(tempConnectionLine);

    console.log('Started connection from', el.dataset.id, position);
}

function updateTempConnection(e) {
    if (!isConnecting || !tempConnectionLine || !connectionStart) return;

    const rect = canvasContainer.getBoundingClientRect();
    const x = (e.clientX - rect.left - panX) / zoom;
    const y = (e.clientY - rect.top - panY) / zoom;

    const from = connectionStart.startPoint;
    const cursor = { x, y, dir: { x: 0, y: 0 } };

    let d;
    if (connectorStyle === 'straight') {
        d = calculateStraightPath(from, cursor);
    } else if (connectorStyle === 'orthogonal') {
        // For temp line, infer a direction toward the cursor
        const dx = x - from.x;
        const dy = y - from.y;
        if (Math.abs(dx) > Math.abs(dy)) {
            cursor.dir = { x: dx > 0 ? -1 : 1, y: 0 };
        } else {
            cursor.dir = { x: 0, y: dy > 0 ? -1 : 1 };
        }
        d = calculateOrthogonalPath(from, cursor);
    } else {
        // Curved — simple bezier to cursor
        const dx = x - from.x;
        const dy = y - from.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const cpDist = Math.min(100, Math.max(30, dist * 0.4));
        const cp1x = from.x + from.dir.x * cpDist;
        const cp1y = from.y + from.dir.y * cpDist;
        d = `M ${from.x} ${from.y} C ${cp1x} ${cp1y}, ${x} ${y}, ${x} ${y}`;
    }
    tempConnectionLine.setAttribute('d', d);
}

function finishConnection(el, position) {
    if (!isConnecting || !connectionStart) return;

    // Remove temp line
    if (tempConnectionLine) {
        tempConnectionLine.remove();
        tempConnectionLine = null;
    }

    if (connectionStart.element !== el) {
        // Determine connection type based on current tool
        const connType = currentTool === 'causal' ? 'causal' : 'arrow';
        createConnection(
            connectionStart.element.dataset.id,
            connectionStart.position,
            el.dataset.id,
            position,
            { type: connType, style: connectorStyle }
        );
        console.log('Created', connType, connectorStyle, 'connection to', el.dataset.id, position);
    }

    isConnecting = false;
    connectionStart = null;
    canvasContainer.style.cursor = (currentTool === 'connect' || currentTool === 'causal') ? 'crosshair' : 'default';
}

function cancelConnection() {
    if (tempConnectionLine) {
        tempConnectionLine.remove();
        tempConnectionLine = null;
    }
    isConnecting = false;
    connectionStart = null;
    canvasContainer.style.cursor = (currentTool === 'connect' || currentTool === 'causal') ? 'crosshair' : 'default';
}

function createConnection(fromId, fromPos, toId, toPos, options = {}, skipHistory = false) {
    if (!skipHistory) saveState();
    const id = `conn-${connections.length}`;
    connections.push({
        id,
        from: { elementId: fromId, position: fromPos },
        to: { elementId: toId, position: toPos },
        ...options
    });
    renderConnections();
}

function renderConnections() {
    // Clear existing paths and labels
    connectionsSvg.querySelectorAll('path, text').forEach(p => p.remove());

    connections.forEach(conn => {
        const fromEl = document.querySelector(`[data-id="${conn.from.elementId}"]`);
        const toEl = document.querySelector(`[data-id="${conn.to.elementId}"]`);

        if (!fromEl || !toEl) return;

        const fromPoint = getConnectionPoint(fromEl, conn.from.position);
        const toPoint = getConnectionPoint(toEl, conn.to.position);

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');

        // Calculate path based on connector style
        const style = conn.style || 'orthogonal';
        let d;
        if (style === 'straight') {
            d = calculateStraightPath(fromPoint, toPoint);
        } else if (style === 'orthogonal') {
            d = calculateOrthogonalPath(fromPoint, toPoint);
        } else {
            d = calculateCurvedPath(fromPoint, toPoint);
        }

        path.setAttribute('d', d);
        path.setAttribute('fill', 'none');
        path.dataset.connectionId = conn.id;

        // Style based on connection type
        const isCausal = conn.type === 'causal';
        if (isCausal) {
            path.setAttribute('stroke', '#e89547');
            path.setAttribute('stroke-width', '3');
            path.setAttribute('marker-end', 'url(#arrowhead-causal)');
            path.classList.add('wb-causal-connection');

            // Add IF label near the start
            const ifLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            ifLabel.setAttribute('x', fromPoint.x + fromPoint.dir.x * 20);
            ifLabel.setAttribute('y', fromPoint.y + fromPoint.dir.y * 20);
            ifLabel.setAttribute('text-anchor', 'middle');
            ifLabel.setAttribute('dominant-baseline', 'middle');
            ifLabel.classList.add('wb-connection-label', 'if-label');
            ifLabel.textContent = 'IF';
            connectionsSvg.appendChild(ifLabel);

            // Add THEN label near the end
            const thenLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            thenLabel.setAttribute('x', toPoint.x - toPoint.dir.x * 25);
            thenLabel.setAttribute('y', toPoint.y - toPoint.dir.y * 25);
            thenLabel.setAttribute('text-anchor', 'middle');
            thenLabel.setAttribute('dominant-baseline', 'middle');
            thenLabel.classList.add('wb-connection-label', 'then-label');
            thenLabel.textContent = 'THEN';
            connectionsSvg.appendChild(thenLabel);
        } else {
            path.setAttribute('stroke', '#4a9f6e');
            path.setAttribute('stroke-width', '2');
            path.setAttribute('marker-end', 'url(#arrowhead)');
        }

        const originalStroke = isCausal ? '#e89547' : '#4a9f6e';
        path.addEventListener('click', () => {
            // Select connection for deletion
            path.setAttribute('stroke', '#e85747');
            setTimeout(() => {
                if (confirm('Delete this connection?')) {
                    saveState();
                    connections = connections.filter(c => c.id !== conn.id);
                    renderConnections();
                } else {
                    path.setAttribute('stroke', originalStroke);
                }
            }, 100);
        });

        connectionsSvg.appendChild(path);
    });
}

function getConnectionPoint(el, position) {
    // Get position directly from element's style (canvas coordinates)
    const x = parseFloat(el.style.left) || 0;
    const y = parseFloat(el.style.top) || 0;
    const w = el.offsetWidth;
    const h = el.offsetHeight;

    // Padding to keep arrows outside shape bounds
    const pad = 8;

    // Diamond vertices extend beyond the bounding box by ~21% of half-width
    const isDiamond = el.querySelector('.wb-process-shape.diamond') || el.classList.contains('diamond');
    const cornerExt = isDiamond ? 0.21 : 0;

    switch (position) {
        case 'top': return { x: x + w/2, y: y - pad - h * cornerExt, dir: { x: 0, y: -1 } };
        case 'bottom': return { x: x + w/2, y: y + h + pad + h * cornerExt, dir: { x: 0, y: 1 } };
        case 'left': return { x: x - pad - w * cornerExt, y: y + h/2, dir: { x: -1, y: 0 } };
        case 'right': return { x: x + w + pad + w * cornerExt, y: y + h/2, dir: { x: 1, y: 0 } };
        default: return { x: x + w/2, y: y + h/2, dir: { x: 0, y: 0 } };
    }
}

// ---------- Straight connector ----------
function calculateStraightPath(from, to) {
    return `M ${from.x} ${from.y} L ${to.x} ${to.y}`;
}

// ---------- Orthogonal (right-angle) connector ----------
function calculateOrthogonalPath(from, to) {
    const stub = 20; // minimum distance out from the port before turning

    // Exit points: go out from the port in its direction
    const ax = from.x + from.dir.x * stub;
    const ay = from.y + from.dir.y * stub;
    const bx = to.x + to.dir.x * stub;
    const by = to.y + to.dir.y * stub;

    const fromH = from.dir.x !== 0; // from port is horizontal
    const toH = to.dir.x !== 0;     // to port is horizontal

    let points;

    if (fromH && toH) {
        // Both horizontal — connect with a vertical bridge
        const midX = (ax + bx) / 2;
        points = [
            { x: from.x, y: from.y },
            { x: ax, y: ay },
            { x: midX, y: ay },
            { x: midX, y: by },
            { x: bx, y: by },
            { x: to.x, y: to.y }
        ];
    } else if (!fromH && !toH) {
        // Both vertical — connect with a horizontal bridge
        const midY = (ay + by) / 2;
        points = [
            { x: from.x, y: from.y },
            { x: ax, y: ay },
            { x: ax, y: midY },
            { x: bx, y: midY },
            { x: bx, y: by },
            { x: to.x, y: to.y }
        ];
    } else {
        // One horizontal, one vertical — single elbow
        if (fromH) {
            // from goes horizontal, to goes vertical → meet at (ax, by)
            // But check if the elbow would backtrack; if so, add extra segment
            const elbowX = bx;
            const elbowY = ay;
            points = [
                { x: from.x, y: from.y },
                { x: ax, y: ay },
                { x: elbowX, y: elbowY },
                { x: bx, y: by },
                { x: to.x, y: to.y }
            ];
        } else {
            // from goes vertical, to goes horizontal → meet at (bx, ay)
            const elbowX = ax;
            const elbowY = by;
            points = [
                { x: from.x, y: from.y },
                { x: ax, y: ay },
                { x: elbowX, y: elbowY },
                { x: bx, y: by },
                { x: to.x, y: to.y }
            ];
        }
    }

    // Remove duplicate consecutive points
    const clean = [points[0]];
    for (let i = 1; i < points.length; i++) {
        const prev = clean[clean.length - 1];
        if (Math.abs(points[i].x - prev.x) > 0.5 || Math.abs(points[i].y - prev.y) > 0.5) {
            clean.push(points[i]);
        }
    }

    return 'M ' + clean.map(p => `${p.x} ${p.y}`).join(' L ');
}

// ---------- Curved (bezier) connector ----------
function calculateCurvedPath(from, to) {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist < 20) {
        return `M ${from.x} ${from.y} L ${to.x} ${to.y}`;
    }

    // Control point distance: proportional to distance, clamped
    const cpDist = Math.min(120, Math.max(30, dist * 0.4));

    // cp1: go out from source in its port direction
    const cp1x = from.x + from.dir.x * cpDist;
    const cp1y = from.y + from.dir.y * cpDist;

    // cp2: approach target from its port direction (to.dir points outward,
    // so adding it pushes the control point AWAY from the target, which
    // makes the curve arrive FROM that direction — exactly what we want)
    const cp2x = to.x + to.dir.x * cpDist;
    const cp2y = to.y + to.dir.y * cpDist;

    return `M ${from.x} ${from.y} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${to.x} ${to.y}`;
}
