// =============================================================================
// Canvas Rendering
// =============================================================================

function renderCanvas() {
    const elLayer = document.getElementById('elements-layer');
    const connLayer = document.getElementById('connections-layer');
    elLayer.innerHTML = '';
    connLayer.innerHTML = '';

    // Render connections
    for (const conn of layout.connections) {
        const fromEl = findElement(conn.from_id);
        const toEl = findElement(conn.to_id);
        if (!fromEl || !toEl) continue;

        const x1 = fromEl.x + 50, y1 = fromEl.y + 25;
        const x2 = toEl.x - 5, y2 = toEl.y + 25;
        const isSelected = selectedElement?.id === conn.id;

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1); line.setAttribute('y1', y1);
        line.setAttribute('x2', x2); line.setAttribute('y2', y2);
        line.setAttribute('class', 'sim-connection' + (isSelected ? ' selected' : ''));
        line.dataset.connId = conn.id;
        // Style by transport type
        const cTT = conn.transport_type || 'none';
        if (cTT === 'walk') { line.setAttribute('stroke-dasharray', '4 3'); line.setAttribute('stroke', '#9ca3af'); }
        else if (cTT === 'hand_cart') { line.setAttribute('stroke-dasharray', '8 3'); line.setAttribute('stroke', '#9ca3af'); }
        else if (cTT === 'forklift') { line.setAttribute('stroke-width', '3'); line.setAttribute('stroke', '#a78bfa'); }
        else if (cTT === 'electric_pj') { line.setAttribute('stroke-width', '2.5'); line.setAttribute('stroke', '#60a5fa'); }
        else if (cTT === 'agv') { line.setAttribute('stroke-width', '2.5'); line.setAttribute('stroke', '#34d399'); line.setAttribute('stroke-dasharray', '6 2'); }
        connLayer.appendChild(line);

        // Wide invisible hit area for clicking
        const hit = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        hit.setAttribute('x1', x1); hit.setAttribute('y1', y1);
        hit.setAttribute('x2', x2); hit.setAttribute('y2', y2);
        hit.setAttribute('class', 'sim-connection-hit');
        hit.onclick = (e) => { e.stopPropagation(); selectElement(conn, 'connection'); };
        connLayer.appendChild(hit);

        // Buffer capacity label
        const mid_x = (x1 + x2) / 2;
        const mid_y = (y1 + y2) / 2 + 15;
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', mid_x);
        label.setAttribute('y', mid_y);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('class', 'state-badge');
        label.textContent = conn.buffer_capacity != null ? `buf:${conn.buffer_capacity}` : '∞';
        label.style.opacity = conn.buffer_capacity != null ? 1 : 0.3;
        label.onclick = (e) => { e.stopPropagation(); selectElement(conn, 'connection'); };
        connLayer.appendChild(label);

        // Transport type label (if set)
        const tType = conn.transport_type || 'none';
        if (tType !== 'none' && conn.transport_distance > 0) {
            const ttDef = TRANSPORT_TYPES[tType] || {};
            const estTime = ttDef.speed ? (conn.transport_distance / ttDef.speed).toFixed(0) : '?';
            const ttG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            ttG.style.cursor = 'pointer';
            ttG.onclick = (e) => { e.stopPropagation(); selectElement(conn, 'connection'); };
            if (ttDef.icon) {
                const iconEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                iconEl.setAttribute('d', ttDef.icon);
                iconEl.setAttribute('transform', `translate(${mid_x - 5}, ${mid_y - 27}) scale(0.42)`);
                iconEl.setAttribute('fill', 'none');
                iconEl.setAttribute('stroke', '#a78bfa');
                iconEl.setAttribute('stroke-width', '2.5');
                iconEl.setAttribute('stroke-linecap', 'round');
                iconEl.setAttribute('stroke-linejoin', 'round');
                ttG.appendChild(iconEl);
            }
            const ttLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            ttLabel.setAttribute('x', mid_x);
            ttLabel.setAttribute('y', mid_y - 14);
            ttLabel.setAttribute('text-anchor', 'middle');
            ttLabel.setAttribute('class', 'transport-badge');
            ttLabel.textContent = `${conn.transport_distance}m ~${estTime}s`;
            ttG.appendChild(ttLabel);
            connLayer.appendChild(ttG);
        }
    }

    // Render sources
    for (const src of layout.sources) {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', `sim-el-source${selectedElement?.id === src.id ? ' selected' : ''}`);
        g.setAttribute('transform', `translate(${src.x}, ${src.y})`);
        g.dataset.elId = src.id;
        g.dataset.elType = 'source';

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', 25); circle.setAttribute('cy', 25); circle.setAttribute('r', 22);
        circle.setAttribute('fill', 'rgba(74,159,110,0.15)');
        circle.setAttribute('stroke', selectedElement?.id === src.id ? '#e8c547' : '#4a9f6e');
        circle.setAttribute('stroke-width', selectedElement?.id === src.id ? 3 : 2);

        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', 25); label.setAttribute('y', 28);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('fill', 'var(--text-primary)');
        label.setAttribute('font-size', '10');
        label.textContent = src.name || 'Source';

        const rateLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        rateLabel.setAttribute('x', 25); rateLabel.setAttribute('y', 60);
        rateLabel.setAttribute('text-anchor', 'middle');
        rateLabel.setAttribute('class', 'state-badge');
        rateLabel.textContent = `${src.arrival_rate || 60}s`;

        g.appendChild(circle);
        g.appendChild(label);
        g.appendChild(rateLabel);

        // Product mix dots under the source
        const pts = src.product_types || [];
        if (pts.length > 1) {
            const dotWidth = Math.min(12, 50 / pts.length);
            const startX = 25 - (pts.length * dotWidth) / 2 + dotWidth / 2;
            pts.forEach((pt, i) => {
                const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                dot.setAttribute('cx', startX + i * dotWidth);
                dot.setAttribute('cy', 70);
                dot.setAttribute('r', 4);
                dot.setAttribute('fill', pt.color || '#4a9f6e');
                dot.setAttribute('stroke', 'var(--bg-card)');
                dot.setAttribute('stroke-width', '1');
                g.appendChild(dot);
            });
        }
        g.onclick = (e) => { e.stopPropagation(); selectElement(src, 'source'); };
        g.onmousedown = (e) => startDrag(e, src);
        elLayer.appendChild(g);
    }

    // Render machines
    for (const stn of layout.stations) {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        const desStn = des ? des.stations.get(stn.id) : null;
        const stateClass = desStn ? ` ${desStn.state}` : '';
        const isBottleneck = desStn && des.getState().bottleneckName === stn.name ? ' bottleneck' : '';
        g.setAttribute('class', `sim-el-machine${selectedElement?.id === stn.id ? ' selected' : ''}${stateClass}${isBottleneck}`);
        g.setAttribute('transform', `translate(${stn.x}, ${stn.y})`);
        g.dataset.elId = stn.id;
        g.dataset.elType = 'machine';

        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('width', 100); rect.setAttribute('height', 50);

        const name = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        name.setAttribute('x', 50); name.setAttribute('y', 20);
        name.setAttribute('text-anchor', 'middle');
        name.setAttribute('fill', 'var(--text-primary)');
        name.setAttribute('font-size', '11');
        name.setAttribute('font-weight', '600');
        name.textContent = stn.name || 'Machine';

        const ct = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        ct.setAttribute('x', 50); ct.setAttribute('y', 35);
        ct.setAttribute('text-anchor', 'middle');
        ct.setAttribute('fill', 'var(--text-secondary)');
        ct.setAttribute('font-size', '9');
        ct.setAttribute('font-family', "'JetBrains Mono', monospace");
        ct.textContent = `CT:${stn.cycle_time || 30}s`;
        if (stn.cycle_time_cv > 0) ct.textContent += ` cv:${(stn.cycle_time_cv * 100).toFixed(0)}%`;

        g.appendChild(rect);
        g.appendChild(name);
        g.appendChild(ct);

        // Queue count badge during sim
        if (desStn && desStn.queue.length > 0) {
            const rushInQueue = desStn.queue.filter(j => j.priority === 'rush' || j.priority === 'hot').length;
            const badge = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            badge.setAttribute('cx', 95); badge.setAttribute('cy', 5);
            badge.setAttribute('r', 8);
            badge.setAttribute('fill', rushInQueue > 0 ? '#dc2626' : desStn.queue.length > 5 ? '#e74c3c' : '#f59e0b');
            const badgeText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            badgeText.setAttribute('x', 95); badgeText.setAttribute('y', 9);
            badgeText.setAttribute('text-anchor', 'middle');
            badgeText.setAttribute('fill', 'white');
            badgeText.setAttribute('font-size', '9');
            badgeText.setAttribute('font-weight', '700');
            badgeText.textContent = desStn.queue.length;
            g.appendChild(badge);
            g.appendChild(badgeText);
            // Rush order indicator — pulsing red dot
            if (rushInQueue > 0) {
                const rushDot = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                rushDot.setAttribute('x', 85); rushDot.setAttribute('y', 9);
                rushDot.setAttribute('font-size', '8');
                rushDot.setAttribute('fill', '#fca5a5');
                rushDot.setAttribute('font-weight', '700');
                rushDot.textContent = `${rushInQueue}!`;
                g.appendChild(rushDot);
            }
        }

        // State indicator
        if (desStn) {
            const stateColors = { idle: '#666', processing: '#4a9f6e', blocked: '#e74c3c', starved: '#f59e0b', down: '#e74c3c', setup: '#8b5cf6', onBreak: '#6366f1' };
            const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            dot.setAttribute('cx', 8); dot.setAttribute('cy', 8);
            dot.setAttribute('r', 5);
            dot.setAttribute('fill', stateColors[desStn.state] || '#666');
            dot.setAttribute('stroke', 'var(--bg-card)');
            dot.setAttribute('stroke-width', '2');
            g.appendChild(dot);

            // Product type indicator — small colored bar at bottom of machine
            if (desStn.currentJob && desStn.currentJob.productColor) {
                const ptBar = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                ptBar.setAttribute('x', 10); ptBar.setAttribute('y', 44);
                ptBar.setAttribute('width', 80); ptBar.setAttribute('height', 4);
                ptBar.setAttribute('rx', 2);
                ptBar.setAttribute('fill', desStn.currentJob.productColor);
                ptBar.setAttribute('opacity', '0.8');
                g.appendChild(ptBar);
            }
        }

        g.onclick = (e) => { e.stopPropagation(); selectElement(stn, 'machine'); };
        g.onmousedown = (e) => startDrag(e, stn);
        elLayer.appendChild(g);
    }

    // Render sinks
    for (const sink of layout.sinks) {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', `sim-el-sink${selectedElement?.id === sink.id ? ' selected' : ''}`);
        g.setAttribute('transform', `translate(${sink.x}, ${sink.y})`);
        g.dataset.elId = sink.id;
        g.dataset.elType = 'sink';

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', 25); circle.setAttribute('cy', 25); circle.setAttribute('r', 22);
        circle.setAttribute('fill', 'rgba(231,76,60,0.15)');
        circle.setAttribute('stroke', selectedElement?.id === sink.id ? '#e8c547' : '#e74c3c');
        circle.setAttribute('stroke-width', selectedElement?.id === sink.id ? 3 : 2);

        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', 25); label.setAttribute('y', 28);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('fill', 'var(--text-primary)');
        label.setAttribute('font-size', '10');
        label.textContent = sink.name || 'Sink';

        // FG warehouse: show inventory levels per product, or completed count for exit sinks
        const desStn = des ? des.stations.get(sink.id) : null;
        const isFG = (sink.sink_mode === 'fg_warehouse');

        if (desStn && isFG && desStn.fgInventory) {
            // Show FG inventory per product type as colored stacked bars
            const entries = Object.entries(desStn.fgInventory).filter(([, v]) => v > 0);
            if (entries.length > 0) {
                const totalFG = entries.reduce((s, [, v]) => s + v, 0);
                let yOff = 55;
                // Find product colors from sources
                const ptColors = {};
                for (const src of layout.sources) {
                    for (const pt of (src.product_types || [])) {
                        ptColors[pt.name] = pt.color;
                    }
                }
                for (const [pt, qty] of entries) {
                    const ptLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    ptLabel.setAttribute('x', 25);
                    ptLabel.setAttribute('y', yOff);
                    ptLabel.setAttribute('text-anchor', 'middle');
                    ptLabel.setAttribute('font-size', '8');
                    ptLabel.setAttribute('font-weight', '600');
                    ptLabel.setAttribute('fill', ptColors[pt] || '#9ca3af');
                    ptLabel.textContent = `${pt}:${qty}`;
                    g.appendChild(ptLabel);
                    yOff += 10;
                }
                // Stockout warning
                const stockouts = Object.values(desStn.stockouts || {}).reduce((s, v) => s + v, 0);
                if (stockouts > 0) {
                    const soLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    soLabel.setAttribute('x', 25);
                    soLabel.setAttribute('y', yOff);
                    soLabel.setAttribute('text-anchor', 'middle');
                    soLabel.setAttribute('font-size', '8');
                    soLabel.setAttribute('font-weight', '700');
                    soLabel.setAttribute('fill', '#e74c3c');
                    soLabel.textContent = `${stockouts} stockouts`;
                    g.appendChild(soLabel);
                }
            }
        } else if (des && des.completedJobs.length > 0) {
            const countLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            countLabel.setAttribute('x', 25); countLabel.setAttribute('y', 60);
            countLabel.setAttribute('text-anchor', 'middle');
            countLabel.setAttribute('fill', 'var(--accent-primary)');
            countLabel.setAttribute('font-size', '11');
            countLabel.setAttribute('font-weight', '700');
            countLabel.textContent = des.completedJobs.length;
            g.appendChild(countLabel);
        }

        // FG warehouse gets a different icon color
        if (isFG) {
            circle.setAttribute('fill', 'rgba(59,130,246,0.15)');
            circle.setAttribute('stroke', selectedElement?.id === sink.id ? '#e8c547' : '#3b82f6');
        }

        g.appendChild(circle);
        g.appendChild(label);
        g.onclick = (e) => { e.stopPropagation(); selectElement(sink, 'sink'); };
        g.onmousedown = (e) => startDrag(e, sink);
        elLayer.appendChild(g);
    }

    updateTransform();
}

function findElement(id) {
    return layout.sources.find(s => s.id === id)
        || layout.stations.find(s => s.id === id)
        || layout.sinks.find(s => s.id === id);
}

function updateTransform() {
    const g = document.getElementById('transform-group');
    g.setAttribute('transform', `translate(${canvasPanX}, ${canvasPanY}) scale(${canvasZoom})`);
    document.getElementById('zoom-display').textContent = `${Math.round(canvasZoom * 100)}%`;
}
