// ============================================================================
// wb-export.js — SVG, PNG, JSON export/import
// ============================================================================

function exportBoard(format) {
    if (format === 'png') {
        // Create a canvas and draw all elements
        const exportCanvas = document.createElement('canvas');
        const ctx = exportCanvas.getContext('2d');

        // Calculate bounds of all elements
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        elements.forEach(el => {
            minX = Math.min(minX, el.x);
            minY = Math.min(minY, el.y);
            const domEl = document.querySelector(`[data-id="${el.id}"]`);
            if (domEl) {
                maxX = Math.max(maxX, el.x + domEl.offsetWidth);
                maxY = Math.max(maxY, el.y + domEl.offsetHeight);
            }
        });

        if (elements.length === 0) {
            alert('Nothing to export. Add some elements first.');
            return;
        }

        const padding = 40;
        const width = maxX - minX + padding * 2;
        const height = maxY - minY + padding * 2;

        exportCanvas.width = width;
        exportCanvas.height = height;

        // Background
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, width, height);

        // Draw grid
        ctx.strokeStyle = '#252525';
        ctx.lineWidth = 1;
        for (let x = 0; x < width; x += 20) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (let y = 0; y < height; y += 20) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Draw connections
        ctx.strokeStyle = '#4a9f6e';
        ctx.lineWidth = 2;
        connections.forEach(conn => {
            const fromEl = document.querySelector(`[data-id="${conn.from.elementId}"]`);
            const toEl = document.querySelector(`[data-id="${conn.to.elementId}"]`);
            if (!fromEl || !toEl) return;

            const from = getConnectionPoint(fromEl, conn.from.position);
            const to = getConnectionPoint(toEl, conn.to.position);

            ctx.beginPath();
            ctx.moveTo(from.x - minX + padding, from.y - minY + padding);
            ctx.lineTo(to.x - minX + padding, to.y - minY + padding);
            ctx.stroke();

            // Arrow
            const angle = Math.atan2(to.y - from.y, to.x - from.x);
            const headLen = 10;
            ctx.beginPath();
            ctx.moveTo(to.x - minX + padding, to.y - minY + padding);
            ctx.lineTo(
                to.x - minX + padding - headLen * Math.cos(angle - Math.PI/6),
                to.y - minY + padding - headLen * Math.sin(angle - Math.PI/6)
            );
            ctx.moveTo(to.x - minX + padding, to.y - minY + padding);
            ctx.lineTo(
                to.x - minX + padding - headLen * Math.cos(angle + Math.PI/6),
                to.y - minY + padding - headLen * Math.sin(angle + Math.PI/6)
            );
            ctx.stroke();
        });

        // Draw elements
        elements.forEach(el => {
            const domEl = document.querySelector(`[data-id="${el.id}"]`);
            if (!domEl) return;

            const x = el.x - minX + padding;
            const y = el.y - minY + padding;
            const w = domEl.offsetWidth;
            const h = domEl.offsetHeight;

            if (el.type === 'postit') {
                const colors = {
                    yellow: '#feff9c', green: '#7afcff', pink: '#ff7eb9',
                    orange: '#ff9f43', blue: '#74b9ff', purple: '#a29bfe'
                };
                ctx.fillStyle = colors[el.color] || '#feff9c';
                ctx.shadowColor = 'rgba(0,0,0,0.3)';
                ctx.shadowBlur = 10;
                ctx.shadowOffsetX = 3;
                ctx.shadowOffsetY = 3;
                ctx.fillRect(x, y, w, h);
                ctx.shadowColor = 'transparent';

                ctx.fillStyle = '#333';
                ctx.font = '13px Inter, sans-serif';
                wrapText(ctx, domEl.textContent, x + 12, y + 20, w - 24, 18);
            } else if (el.type === 'group') {
                ctx.strokeStyle = '#4a9f6e';
                ctx.setLineDash([8, 4]);
                ctx.lineWidth = 2;
                ctx.strokeRect(x, y, w, h);
                ctx.setLineDash([]);

                ctx.fillStyle = '#4a9f6e';
                ctx.fillRect(x + 8, y + 8, w - 16, 30);
                ctx.fillStyle = 'white';
                ctx.font = '600 14px Inter, sans-serif';
                ctx.fillText(el.title || 'Group', x + 16, y + 28);
            } else if (el.type === 'text') {
                ctx.fillStyle = '#e8efe8';
                ctx.font = '14px Inter, sans-serif';
                ctx.fillText(domEl.textContent, x, y + 14);
            } else if (el.type === 'fishbone') {
                // Draw fishbone diagram
                const spineY = y + h/2;

                // Main spine
                ctx.strokeStyle = '#4a9f6e';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(x + 50, spineY);
                ctx.lineTo(x + w - 120, spineY);
                ctx.stroke();

                // Effect box
                ctx.fillStyle = '#4a9f6e';
                ctx.fillRect(x + w - 120, spineY - 30, 120, 60);
                ctx.fillStyle = 'white';
                ctx.font = '600 14px Inter, sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(el.effect || 'Effect', x + w - 60, spineY + 5);
                ctx.textAlign = 'left';

                // Draw categories and causes
                if (el.categories) {
                    const boneSpacing = (w - 250) / 3;
                    el.categories.forEach((cat, i) => {
                        const isTop = i < 3;
                        const boneX = x + 80 + (i % 3) * boneSpacing;

                        // Bone line
                        ctx.strokeStyle = '#4a9f6e';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(boneX, spineY);
                        const endY = isTop ? spineY - 100 : spineY + 100;
                        ctx.lineTo(boneX + 30, endY);
                        ctx.stroke();

                        // Category label
                        ctx.fillStyle = '#4a9f6e';
                        ctx.font = '600 12px Inter, sans-serif';
                        const catY = isTop ? spineY - 115 : spineY + 125;
                        ctx.fillText(cat.name, boneX, catY);

                        // Causes
                        ctx.font = '11px Inter, sans-serif';
                        ctx.fillStyle = '#e8efe8';
                        cat.causes.forEach((cause, j) => {
                            const causeY = isTop
                                ? catY - 18 - j * 16
                                : catY + 18 + j * 16;
                            ctx.fillText('• ' + cause, boneX, causeY);
                        });
                    });
                }
            } else if (el.type === 'image' && el.src) {
                // Images drawn in second pass (see below)
            } else {
                // Process shapes
                ctx.fillStyle = '#121a12';
                ctx.strokeStyle = '#4a9f6e';
                ctx.lineWidth = 2;

                if (el.type === 'diamond') {
                    ctx.beginPath();
                    ctx.moveTo(x + w/2, y);
                    ctx.lineTo(x + w, y + h/2);
                    ctx.lineTo(x + w/2, y + h);
                    ctx.lineTo(x, y + h/2);
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();
                } else if (el.type === 'oval') {
                    ctx.beginPath();
                    ctx.ellipse(x + w/2, y + h/2, w/2, h/2, 0, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();
                } else {
                    ctx.fillRect(x, y, w, h);
                    ctx.strokeRect(x, y, w, h);
                }

                ctx.fillStyle = '#e8efe8';
                ctx.font = '12px Inter, sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(el.text || '', x + w/2, y + h/2 + 4);
                ctx.textAlign = 'left';
            }
        });

        // Second pass: draw images (preload them first)
        const imageEls = elements.filter(el => el.type === 'image' && el.src);
        let imagesLoaded = 0;
        const totalImages = imageEls.length;

        function finishExport() {
            const link = document.createElement('a');
            link.download = 'whiteboard.png';
            link.href = exportCanvas.toDataURL('image/png');
            link.click();
        }

        if (totalImages === 0) {
            finishExport();
        } else {
            imageEls.forEach(el => {
                const domEl = document.querySelector(`[data-id="${el.id}"]`);
                if (!domEl) { imagesLoaded++; return; }
                const imgObj = new Image();
                imgObj.onload = () => {
                    const ix = el.x - minX + padding;
                    const iy = el.y - minY + padding;
                    ctx.drawImage(imgObj, ix, iy, domEl.offsetWidth, domEl.offsetHeight);
                    imagesLoaded++;
                    if (imagesLoaded >= totalImages) finishExport();
                };
                imgObj.onerror = () => { imagesLoaded++; if (imagesLoaded >= totalImages) finishExport(); };
                imgObj.src = el.src;
            });
        }
        return; // Exit early since download is async now
    }
}

function wrapText(ctx, text, x, y, maxWidth, lineHeight) {
    const words = text.split(' ');
    let line = '';

    for (let n = 0; n < words.length; n++) {
        const testLine = line + words[n] + ' ';
        const metrics = ctx.measureText(testLine);
        if (metrics.width > maxWidth && n > 0) {
            ctx.fillText(line, x, y);
            line = words[n] + ' ';
            y += lineHeight;
        } else {
            line = testLine;
        }
    }
    ctx.fillText(line, x, y);
}

// ============================================================================
// SVG Export
// ============================================================================

function exportBoardSVG() {
    if (elements.length === 0) {
        alert('Nothing to export. Add some elements first.');
        return;
    }

    // Calculate bounds
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    elements.forEach(el => {
        const domEl = document.querySelector(`[data-id="${el.id}"]`);
        if (!domEl) return;
        minX = Math.min(minX, el.x);
        minY = Math.min(minY, el.y);
        maxX = Math.max(maxX, el.x + domEl.offsetWidth);
        maxY = Math.max(maxY, el.y + domEl.offsetHeight);
    });

    const pad = 40;
    const width = maxX - minX + pad * 2;
    const height = maxY - minY + pad * 2;
    const ox = -minX + pad; // offset x
    const oy = -minY + pad; // offset y

    let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">\n`;
    svg += `<rect width="${width}" height="${height}" fill="#1a1a1a"/>\n`;

    // Defs for arrowheads
    svg += `<defs>
  <marker id="ah" markerWidth="12" markerHeight="8" refX="11" refY="4" orient="auto"><polygon points="0 0,12 4,0 8" fill="#4a9f6e"/></marker>
  <marker id="ahc" markerWidth="14" markerHeight="10" refX="13" refY="5" orient="auto"><polygon points="0 0,14 5,0 10" fill="#e89547"/></marker>
</defs>\n`;

    // Connections
    connections.forEach(conn => {
        const fromEl = document.querySelector(`[data-id="${conn.from.elementId}"]`);
        const toEl = document.querySelector(`[data-id="${conn.to.elementId}"]`);
        if (!fromEl || !toEl) return;
        const from = getConnectionPoint(fromEl, conn.from.position);
        const to = getConnectionPoint(toEl, conn.to.position);
        const f = { x: from.x + ox, y: from.y + oy, dir: from.dir };
        const t = { x: to.x + ox, y: to.y + oy, dir: to.dir };

        let pathD;
        if (conn.style === 'straight') pathD = calculateStraightPath(f, t);
        else if (conn.style === 'curved') pathD = calculateCurvedPath(f, t);
        else pathD = calculateOrthogonalPath(f, t);

        const isCausal = conn.type === 'causal';
        const color = isCausal ? '#e89547' : '#4a9f6e';
        const sw = isCausal ? 3 : 2;
        const marker = isCausal ? 'ahc' : 'ah';
        svg += `<path d="${pathD}" fill="none" stroke="${color}" stroke-width="${sw}" marker-end="url(#${marker})"/>\n`;

        if (isCausal) {
            const mx = (f.x + t.x) / 2, my = (f.y + t.y) / 2;
            svg += `<text x="${f.x + (t.x - f.x) * 0.2}" y="${f.y + (t.y - f.y) * 0.2 - 8}" fill="#e89547" font-size="11" font-family="Inter,sans-serif" font-weight="600">IF</text>\n`;
            svg += `<text x="${f.x + (t.x - f.x) * 0.8}" y="${f.y + (t.y - f.y) * 0.8 - 8}" fill="#e89547" font-size="11" font-family="Inter,sans-serif" font-weight="600">THEN</text>\n`;
        }
    });

    // Elements
    const postitColors = { yellow: '#feff9c', green: '#7afcff', pink: '#ff7eb9', orange: '#ff9f43', blue: '#74b9ff', purple: '#a29bfe' };

    elements.forEach(el => {
        const domEl = document.querySelector(`[data-id="${el.id}"]`);
        if (!domEl) return;
        const ex = el.x + ox;
        const ey = el.y + oy;
        const w = domEl.offsetWidth;
        const h = domEl.offsetHeight;

        if (el.type === 'postit') {
            const col = postitColors[el.color] || '#feff9c';
            svg += `<rect x="${ex}" y="${ey}" width="${w}" height="${h}" fill="${col}" rx="2"/>\n`;
            svg += `<foreignObject x="${ex}" y="${ey}" width="${w}" height="${h}"><div xmlns="http://www.w3.org/1999/xhtml" style="padding:12px;font-size:13px;color:#333;font-family:Inter,sans-serif;word-wrap:break-word;">${escapeHtml(domEl.textContent)}</div></foreignObject>\n`;
        } else if (el.type === 'text') {
            svg += `<text x="${ex}" y="${ey + 16}" fill="#e8efe8" font-size="14" font-family="Inter,sans-serif">${escapeHtml(domEl.textContent)}</text>\n`;
        } else if (el.type === 'group') {
            svg += `<rect x="${ex}" y="${ey}" width="${w}" height="${h}" fill="rgba(74,159,110,0.1)" stroke="#4a9f6e" stroke-width="2" stroke-dasharray="8,4" rx="8"/>\n`;
            const title = el.title || 'Group';
            svg += `<rect x="${ex + 8}" y="${ey + 8}" width="${w - 16}" height="32" fill="#4a9f6e" rx="4"/>\n`;
            svg += `<text x="${ex + 20}" y="${ey + 28}" fill="white" font-size="14" font-weight="600" font-family="Inter,sans-serif">${escapeHtml(title)}</text>\n`;
        } else if (el.type === 'rectangle') {
            svg += `<rect x="${ex}" y="${ey}" width="${w}" height="${h}" fill="#121a12" stroke="#4a9f6e" stroke-width="2" rx="4"/>\n`;
            svg += `<text x="${ex + w/2}" y="${ey + h/2 + 5}" fill="#e8efe8" font-size="12" text-anchor="middle" font-family="Inter,sans-serif">${escapeHtml(el.text || '')}</text>\n`;
        } else if (el.type === 'diamond') {
            const cx = ex + w/2, cy = ey + h/2;
            const hw = w/2, hh = h/2;
            svg += `<polygon points="${cx},${cy-hh} ${cx+hw},${cy} ${cx},${cy+hh} ${cx-hw},${cy}" fill="#121a12" stroke="#4a9f6e" stroke-width="2"/>\n`;
            svg += `<text x="${cx}" y="${cy + 5}" fill="#e8efe8" font-size="12" text-anchor="middle" font-family="Inter,sans-serif">${escapeHtml(el.text || '')}</text>\n`;
        } else if (el.type === 'oval') {
            svg += `<ellipse cx="${ex + w/2}" cy="${ey + h/2}" rx="${w/2}" ry="${h/2}" fill="#121a12" stroke="#4a9f6e" stroke-width="2"/>\n`;
            svg += `<text x="${ex + w/2}" y="${ey + h/2 + 5}" fill="#e8efe8" font-size="12" text-anchor="middle" font-family="Inter,sans-serif">${escapeHtml(el.text || '')}</text>\n`;
        } else if (el.type === 'parallelogram') {
            const skew = w * 0.15;
            svg += `<polygon points="${ex+skew},${ey} ${ex+w},${ey} ${ex+w-skew},${ey+h} ${ex},${ey+h}" fill="#121a12" stroke="#4a9f6e" stroke-width="2"/>\n`;
            svg += `<text x="${ex + w/2}" y="${ey + h/2 + 5}" fill="#e8efe8" font-size="12" text-anchor="middle" font-family="Inter,sans-serif">${escapeHtml(el.text || '')}</text>\n`;
        } else if (el.type === 'document' || el.type === 'cylinder') {
            svg += `<rect x="${ex}" y="${ey}" width="${w}" height="${h}" fill="#121a12" stroke="#4a9f6e" stroke-width="2" rx="2"/>\n`;
            svg += `<text x="${ex + w/2}" y="${ey + h/2 + 5}" fill="#e8efe8" font-size="12" text-anchor="middle" font-family="Inter,sans-serif">${escapeHtml(el.text || '')}</text>\n`;
        } else if (el.type === 'image' && el.src) {
            svg += `<image href="${el.src}" x="${ex}" y="${ey}" width="${w}" height="${h}" preserveAspectRatio="xMidYMid meet"/>\n`;
        } else if (el.type === 'gate-and') {
            svg += `<rect x="${ex}" y="${ey}" width="${w}" height="${h}" fill="#121a12" stroke="#8a7fbf" stroke-width="2" rx="${h/2}"/>\n`;
            svg += `<text x="${ex + w/2}" y="${ey + h/2 + 5}" fill="#8a7fbf" font-size="11" text-anchor="middle" font-weight="700" font-family="Inter,sans-serif">AND</text>\n`;
        } else if (el.type === 'gate-or') {
            svg += `<rect x="${ex}" y="${ey}" width="${w}" height="${h}" fill="#121a12" stroke="#ff7eb9" stroke-width="2" rx="${h/2}"/>\n`;
            svg += `<text x="${ex + w/2}" y="${ey + h/2 + 5}" fill="#ff7eb9" font-size="11" text-anchor="middle" font-weight="700" font-family="Inter,sans-serif">OR</text>\n`;
        }
    });

    svg += '</svg>';

    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `whiteboard-${Date.now()}.svg`;
    a.click();
    URL.revokeObjectURL(url);
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ============================================================================
// JSON Export/Import
// ============================================================================

function exportBoardJSON() {
    const boardData = {
        elements,
        connections,
        zoom,
        panX,
        panY,
        timestamp: new Date().toISOString(),
        version: '1.0'
    };

    const json = JSON.stringify(boardData, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const link = document.createElement('a');
    link.download = `whiteboard-${Date.now()}.json`;
    link.href = URL.createObjectURL(blob);
    link.click();
    URL.revokeObjectURL(link.href);
}

// ============================================================================
// Causal Relationship Export (If-Then → Hypotheses)
// ============================================================================

function getCausalRelationships() {
    // Find all causal connections
    const causalConns = connections.filter(c => c.type === 'causal');

    // Build hypothesis candidates
    return causalConns.map(conn => {
        const fromEl = elements.find(e => e.id === conn.from.elementId);
        const toEl = elements.find(e => e.id === conn.to.elementId);

        if (!fromEl || !toEl) return null;

        const condition = fromEl.text || fromEl.title || fromEl.effect || 'Unknown condition';
        const effect = toEl.text || toEl.title || toEl.effect || 'Unknown effect';

        // Check if connected through AND/OR gates
        const isGate = fromEl.type === 'gate-and' || fromEl.type === 'gate-or';

        return {
            id: conn.id,
            statement: `If ${condition}, then ${effect}`,
            condition,
            effect,
            conditionElementId: fromEl.id,
            effectElementId: toEl.id,
            hasGate: isGate,
            gateType: isGate ? fromEl.gateType : null
        };
    }).filter(Boolean);
}

async function exportCausalAsHypotheses() {
    const causalRels = getCausalRelationships();

    if (causalRels.length === 0) {
        alert('No if-then relationships found. Use the If-Then tool (I) to create causal connections.');
        return;
    }

    // Check if board is collaborative (has room code)
    if (!roomCode) {
        // Local-only mode - just show the data
        const summary = causalRels.map((rel, i) => `${i + 1}. ${rel.statement}`).join('\n');
        alert(`Found ${causalRels.length} hypothesis candidate(s):\n\n${summary}\n\n(Save board to a project to export to hypotheses)`);
        console.log('Causal relationships (local):', causalRels);
        return causalRels;
    }

    // Format for API
    const hypotheses = causalRels.map(rel => ({
        statement: rel.statement,
        condition: rel.condition,
        effect: rel.effect,
        prior_probability: 0.5,
    }));

    // Show confirmation
    const summary = hypotheses.map((h, i) => `${i + 1}. ${h.statement}`).join('\n');
    if (!confirm(`Export ${hypotheses.length} hypothesis candidate(s) to project?\n\n${summary}`)) {
        return;
    }

    try {
        const response = await fetch(`/api/whiteboard/boards/${roomCode}/export-hypotheses/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ causal_relationships: hypotheses })
        });

        const data = await response.json();

        if (!response.ok) {
            alert(`Error: ${safeStr(data.error, 'Failed to export hypotheses')}`);
            return;
        }

        // Show success
        let message = `Exported to project "${data.project_title}":\n`;
        message += `- ${data.created_count} new hypothesis(es) created\n`;
        if (data.existing_count > 0) {
            message += `- ${data.existing_count} already existed (skipped)`;
        }
        alert(message);

        console.log('Export result:', data);
        return data;

    } catch (err) {
        console.error('Export failed:', err);
        alert('Failed to export hypotheses. Check console for details.');
    }
}

function importBoardJSON(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const data = JSON.parse(e.target.result);

            if (!data.elements || !Array.isArray(data.elements)) {
                alert('Invalid whiteboard file format.');
                return;
            }

            if (elements.length > 0 && !confirm('This will replace your current board. Continue?')) {
                return;
            }

            clearCanvas();
            connections = data.connections || [];
            zoom = data.zoom || 1;
            panX = data.panX || 0;
            panY = data.panY || 0;

            updateCanvasTransform();
            document.getElementById('zoom-display').textContent = `${Math.round(zoom * 100)}%`;

            // Recreate elements
            data.elements.forEach(el => {
                createElement(el.type, el.x, el.y, el);
                elementIdCounter = Math.max(elementIdCounter, parseInt(el.id.split('-')[1]) || 0);
            });

            // Recreate connections
            setTimeout(() => renderConnections(), 100);

            alert('Board imported successfully!');
        } catch (err) {
            alert('Error importing file: ' + safeStr(err, 'Unknown error'));
        }
    };
    reader.readAsText(file);

    // Reset input so same file can be imported again
    event.target.value = '';
}
