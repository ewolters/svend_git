/**
 * VSM Canvas Module
 * All SVG rendering — process boxes, inventory, connections, lead time ladder.
 * MIGRATION: Extracted from templates/vsm.html
 */


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


function formatTime(seconds) {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${(seconds/60).toFixed(1)}m`;
    return `${(seconds/3600).toFixed(1)}h`;
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

    // Double-click to edit
    g.addEventListener('dblclick', () => showEntityProperties(ent, entityType));

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
    g.addEventListener('dblclick', () => showInventoryProperties(inv));

    layer.appendChild(g);
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
    text.textContent = inv.days_of_supply ? `${inv.days_of_supply}d` : inv.quantity || 'I';
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
    g.addEventListener('dblclick', () => showInventoryProperties(inv));

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
    g.addEventListener('dblclick', () => showKaizenProperties(burst));

    layer.appendChild(g);
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
        const waitDays = inv.days_of_supply || 0;
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
    g.addEventListener('dblclick', () => {
        clearTimeout(stepMetricsClickTimer);
        closeStepMetrics();
        showProperties(step);
    });

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
    g.addEventListener('dblclick', () => showInventoryProperties(inv));

    layer.appendChild(g);
}

// =============================================================================
function renderVSM() {
    if (!currentVSM) return;

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
    g.addEventListener('dblclick', () => showWorkCenterProperties(wc));

    layer.appendChild(g);
}

