/**
 * calc-qfd.js — House of Quality (QFD) Calculator
 *
 * Load order: after calc-core.js (uses SvendOps, renderNextSteps)
 * Extracted from: calculators.html (inline script)
 *
 * Contains: QFD matrix, correlation matrix, competitive benchmarking
 */

// ============================================================================
// House of Quality (QFD)
// ============================================================================

let qfdCurrentPhase = 1;

// Phase 1 data
let qfdWhats = [
    { name: 'Easy to use', importance: 5 },
    { name: 'Durable', importance: 4 },
    { name: 'Lightweight', importance: 3 },
];

let qfdHows = [
    { name: 'Material strength', unit: 'MPa', target: '' },
    { name: 'Weight', unit: 'kg', target: '' },
    { name: 'Ergonomic score', unit: '1-10', target: '' },
];

let qfdRelationships = {}; // { 'what_idx-how_idx': strength (0,1,3,9) }
let qfdCorrelations = {}; // { 'how1_idx-how2_idx': correlation (++,+,0,-,--) }
let qfdWhatCorrelations = {}; // { 'what1_idx-what2_idx': '0', '?', '!' }

// Phase 2-4 data
let qfdParts = [];
let qfdProcesses = [];
let qfdControls = [];
let qfdMatrix2 = {};
let qfdMatrix3 = {};
let qfdMatrix4 = {};

function setQFDPhase(phase) {
    qfdCurrentPhase = phase;

    // Update tabs
    for (let i = 1; i <= 4; i++) {
        const tab = document.getElementById(`qfd-tab-${i}`);
        const panel = document.getElementById(`qfd-phase-${i}`);
        if (tab && panel) {
            if (i === phase) {
                tab.style.borderBottomColor = 'var(--accent)';
                tab.style.color = 'var(--accent)';
                tab.style.fontWeight = '600';
                panel.style.display = 'block';
            } else {
                tab.style.borderBottomColor = 'transparent';
                tab.style.color = 'var(--text-dim)';
                tab.style.fontWeight = 'normal';
                panel.style.display = 'none';
            }
        }
    }

    // Render appropriate matrix
    if (phase === 1) renderQFDMatrix();
    else if (phase === 2) renderQFDMatrix2();
    else if (phase === 3) renderQFDMatrix3();
    else if (phase === 4) renderQFDMatrix4();
}

function renderQFDWhats() {
    const container = document.getElementById('qfd-whats');
    if (!container) return;

    container.innerHTML = qfdWhats.map((w, i) => `
        <div class="yamazumi-station" style="gap: 8px;">
            <input type="text" value="${w.name}" style="flex: 1; padding: 8px;"
                   oninput="updateQFDWhat(${i}, 'name', this.value)" placeholder="Customer requirement">
            <input type="number" value="${w.importance}" min="1" max="5" style="width: 50px; text-align: center;"
                   oninput="updateQFDWhat(${i}, 'importance', this.value)" title="Importance (1-5)">
            <span style="color: var(--text-dim); font-size: 11px;">imp</span>
            <button class="yamazumi-station-remove" onclick="removeQFDWhat(${i})">&times;</button>
        </div>
    `).join('');

    renderQFDMatrix();
}

function renderQFDHows() {
    const container = document.getElementById('qfd-hows');
    if (!container) return;

    container.innerHTML = qfdHows.map((h, i) => `
        <div class="yamazumi-station" style="gap: 8px;">
            <input type="text" value="${h.name}" style="flex: 1; padding: 8px;"
                   oninput="updateQFDHow(${i}, 'name', this.value)" placeholder="Engineering characteristic">
            <input type="text" value="${h.unit}" style="width: 60px; text-align: center;"
                   oninput="updateQFDHow(${i}, 'unit', this.value)" placeholder="unit">
            <input type="text" value="${h.target}" style="width: 70px; text-align: center;"
                   oninput="updateQFDHow(${i}, 'target', this.value)" placeholder="target">
            <button class="yamazumi-station-remove" onclick="removeQFDHow(${i})">&times;</button>
        </div>
    `).join('');

    renderQFDMatrix();
}

function addQFDWhat() {
    qfdWhats.push({ name: 'New requirement', importance: 3 });
    renderQFDWhats();
}

function removeQFDWhat(idx) {
    qfdWhats.splice(idx, 1);
    // Clean up relationships
    Object.keys(qfdRelationships).forEach(key => {
        if (key.startsWith(`${idx}-`)) delete qfdRelationships[key];
    });
    renderQFDWhats();
}

function updateQFDWhat(idx, field, value) {
    if (field === 'importance') value = parseInt(value) || 3;
    qfdWhats[idx][field] = value;
    renderQFDMatrix();
}

function addQFDHow() {
    qfdHows.push({ name: 'New characteristic', unit: '', target: '' });
    renderQFDHows();
}

function removeQFDHow(idx) {
    qfdHows.splice(idx, 1);
    // Clean up relationships and correlations
    Object.keys(qfdRelationships).forEach(key => {
        if (key.endsWith(`-${idx}`)) delete qfdRelationships[key];
    });
    Object.keys(qfdCorrelations).forEach(key => {
        if (key.includes(`${idx}`)) delete qfdCorrelations[key];
    });
    renderQFDHows();
}

function updateQFDHow(idx, field, value) {
    qfdHows[idx][field] = value;
    renderQFDMatrix();
}

function updateQFDRelationship(whatIdx, howIdx, value) {
    const key = `${whatIdx}-${howIdx}`;
    qfdRelationships[key] = parseInt(value) || 0;
    renderQFDMatrix();
}

function updateQFDCorrelation(how1Idx, how2Idx, value) {
    const key = `${Math.min(how1Idx, how2Idx)}-${Math.max(how1Idx, how2Idx)}`;
    qfdCorrelations[key] = value;
    renderQFDMatrix();
}

function renderQFDMatrix() {
    const container = document.getElementById('qfd-matrix');
    if (!container) return;

    if (qfdWhats.length === 0 || qfdHows.length === 0) {
        container.innerHTML = '<div style="padding: 20px; color: var(--text-dim); text-align: center;">Add customer requirements and engineering characteristics to build the matrix</div>';
        return;
    }

    const howCount = qfdHows.length;
    const symbols = { 9: '●', 3: '○', 1: '△', 0: '' };
    const corrSymbols = { '++': '++', '+': '+', '0': '', '-': '−', '--': '−−' };
    const corrColors = { '++': '#4a9f6e', '+': '#7bc47f', '0': '#666', '-': '#e8a87c', '--': '#e74c3c' };

    let html = '<div style="position: relative;">';


    // Main matrix table
    html += '<table style="border-collapse: collapse; width: 100%;">';

    // Header row (HOWs)
    html += '<tr><th style="width: 150px; padding: 8px; background: var(--bg-card); border: 1px solid var(--border);"></th>';
    qfdHows.forEach((h, i) => {
        html += `<th style="padding: 8px; background: var(--bg-secondary); border: 1px solid var(--border); min-width: 50px; writing-mode: vertical-lr; text-orientation: mixed; height: 100px; font-size: 11px; font-weight: 500;">${h.name}</th>`;
    });
    html += '<th style="width: 60px; padding: 8px; background: var(--bg-card); border: 1px solid var(--border); font-size: 10px;">Weight</th></tr>';

    // Data rows (WHATs)
    qfdWhats.forEach((w, wi) => {
        html += '<tr>';
        html += `<td style="padding: 8px; background: var(--bg-secondary); border: 1px solid var(--border); font-size: 12px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: var(--text-dim); font-size: 10px;">${w.importance}</span>
                ${w.name}
            </div>
        </td>`;

        let rowWeight = 0;
        qfdHows.forEach((h, hi) => {
            const key = `${wi}-${hi}`;
            const rel = qfdRelationships[key] || 0;
            rowWeight += rel * w.importance;

            html += `<td style="padding: 4px; border: 1px solid var(--border); text-align: center; cursor: pointer; background: ${rel === 9 ? 'rgba(74, 159, 110, 0.2)' : rel === 3 ? 'rgba(74, 159, 110, 0.1)' : 'transparent'};" onclick="cycleRelationship(${wi}, ${hi})">
                <span style="font-size: 16px;">${symbols[rel]}</span>
            </td>`;
        });

        html += `<td style="padding: 8px; border: 1px solid var(--border); text-align: center; font-size: 11px; color: var(--text-dim);">${rowWeight}</td>`;
        html += '</tr>';
    });

    // Column totals
    html += '<tr><td style="padding: 8px; background: var(--bg-card); border: 1px solid var(--border); font-size: 11px; font-weight: 600;">Priority Score</td>';
    const priorities = [];
    qfdHows.forEach((h, hi) => {
        let score = 0;
        qfdWhats.forEach((w, wi) => {
            const key = `${wi}-${hi}`;
            score += (qfdRelationships[key] || 0) * w.importance;
        });
        priorities.push({ name: h.name, score });
        html += `<td style="padding: 8px; background: var(--accent); color: white; border: 1px solid var(--border); text-align: center; font-weight: 600;">${score}</td>`;
    });
    html += '<td style="border: 1px solid var(--border);"></td></tr>';

    // Target row
    html += '<tr><td style="padding: 8px; background: var(--bg-card); border: 1px solid var(--border); font-size: 11px;">Target</td>';
    qfdHows.forEach((h, hi) => {
        html += `<td style="padding: 4px; border: 1px solid var(--border); text-align: center; font-size: 10px; color: var(--text-dim);">${h.target || '—'}</td>`;
    });
    html += '<td style="border: 1px solid var(--border);"></td></tr>';

    html += '</table></div>';

    // Legend
    html += `
        <div style="display: flex; gap: 20px; margin-top: 16px; font-size: 11px; color: var(--text-dim); flex-wrap: wrap;">
            <span>Relationships: <strong>●</strong>=Strong(9) <strong>○</strong>=Medium(3) <strong>△</strong>=Weak(1)</span>
            <span style="color: var(--text-secondary);">Click cells to cycle values</span>
        </div>
    `;

    // Correlation Matrices side by side
    const whatCount = qfdWhats.length;
    html += '<div style="display: flex; gap: 24px; margin-top: 24px; flex-wrap: wrap; align-items: flex-start;">';

    // WHAT × WHAT Correlation Matrix (requirement conflicts)
    if (whatCount > 1) {
        html += `
            <div style="flex: 1; min-width: 280px;">
                <div style="font-size: 13px; font-weight: 600; margin-bottom: 12px; color: var(--text-secondary);">WHAT × WHAT Conflicts</div>
                <table style="border-collapse: collapse;">
                    <tr><th style="padding: 6px; background: var(--bg-card); border: 1px solid var(--border);"></th>`;

        qfdWhats.forEach((w, i) => {
            html += '<th style="padding: 6px 8px; background: var(--bg-secondary); border: 1px solid var(--border); font-size: 10px; max-width: 80px; overflow: hidden; text-overflow: ellipsis;">' + w.name + '</th>';
        });
        html += '</tr>';

        qfdWhats.forEach((w1, i) => {
            html += '<tr><td style="padding: 6px 8px; background: var(--bg-secondary); border: 1px solid var(--border); font-size: 10px; white-space: nowrap;">' + w1.name + '</td>';
            qfdWhats.forEach((w2, j) => {
                if (i === j) {
                    html += '<td style="padding: 6px; border: 1px solid var(--border); background: var(--bg-card); text-align: center;">—</td>';
                } else if (i > j) {
                    // Mirror the upper triangle
                    const key = j + '-' + i;
                    const corr = qfdWhatCorrelations[key] || '0';
                    const bgColor = corr === '!' ? 'rgba(231, 76, 60, 0.3)' :
                                   corr === '?' ? 'rgba(232, 197, 71, 0.2)' : 'transparent';
                    html += '<td style="padding: 6px; border: 1px solid var(--border); text-align: center; background: ' + bgColor + '; color: var(--text-dim);">' + (corr === '0' ? '·' : corr) + '</td>';
                } else {
                    const key = i + '-' + j;
                    const corr = qfdWhatCorrelations[key] || '0';
                    const bgColor = corr === '!' ? 'rgba(231, 76, 60, 0.3)' :
                                   corr === '?' ? 'rgba(232, 197, 71, 0.2)' : 'transparent';
                    html += '<td style="padding: 6px; border: 1px solid var(--border); text-align: center; cursor: pointer; background: ' + bgColor + '; font-weight: bold; color: ' + (corr === '!' ? '#e74c3c' : corr === '?' ? '#e8c547' : 'var(--text-dim)') + ';" onclick="cycleWhatCorrelation(' + i + ', ' + j + ')">' + (corr === '0' ? '·' : corr) + '</td>';
                }
            });
            html += '</tr>';
        });

        html += '</table>';

        // Count conflicts
        let conflictCount = 0;
        let tensionCount = 0;
        Object.values(qfdWhatCorrelations).forEach(v => {
            if (v === '!') conflictCount++;
            if (v === '?') tensionCount++;
        });

        if (conflictCount > 0) {
            html += '<div style="margin-top: 8px; padding: 8px 12px; background: rgba(231, 76, 60, 0.1); border-left: 3px solid #e74c3c; border-radius: 4px; font-size: 12px; color: #e74c3c;"><strong>' + conflictCount + ' conflict(s)</strong> — requirements may be mutually exclusive. Discuss with stakeholders.</div>';
        } else if (tensionCount > 0) {
            html += '<div style="margin-top: 8px; padding: 8px 12px; background: rgba(232, 197, 71, 0.1); border-left: 3px solid #e8c547; border-radius: 4px; font-size: 12px; color: #b8860b;">' + tensionCount + ' tension(s) — requirements may trade off. Document assumptions.</div>';
        }

        html += '<div style="margin-top: 8px; font-size: 10px; color: var(--text-dim);"><strong style="color:#e74c3c">!</strong> Conflict (mutually exclusive) &nbsp; <strong style="color:#e8c547">?</strong> Tension (tradeoff likely)</div></div>';
    }

    // HOW × HOW Correlation Matrix
    if (howCount > 1) {
        html += `
            <div style="flex: 1; min-width: 280px;">
                <div style="font-size: 13px; font-weight: 600; margin-bottom: 12px; color: var(--text-secondary);">HOW × HOW Correlations</div>
                <table style="border-collapse: collapse;">
                    <tr><th style="padding: 6px; background: var(--bg-card); border: 1px solid var(--border);"></th>`;

        qfdHows.forEach((h, i) => {
            html += `<th style="padding: 6px 8px; background: var(--bg-secondary); border: 1px solid var(--border); font-size: 10px; max-width: 80px; overflow: hidden; text-overflow: ellipsis;">${h.name}</th>`;
        });
        html += '</tr>';

        qfdHows.forEach((h1, i) => {
            html += '<tr><td style="padding: 6px 8px; background: var(--bg-secondary); border: 1px solid var(--border); font-size: 10px; white-space: nowrap;">' + h1.name + '</td>';
            qfdHows.forEach((h2, j) => {
                if (i === j) {
                    html += '<td style="padding: 6px; border: 1px solid var(--border); background: var(--bg-card); text-align: center;">—</td>';
                } else {
                    const key = i < j ? (i + '-' + j) : (j + '-' + i);
                    const corr = qfdCorrelations[key] || '0';
                    const bgColor = corr === '++' ? 'rgba(74, 159, 110, 0.3)' :
                                   corr === '+' ? 'rgba(74, 159, 110, 0.15)' :
                                   corr === '--' ? 'rgba(231, 76, 60, 0.3)' :
                                   corr === '-' ? 'rgba(231, 76, 60, 0.15)' : 'transparent';
                    html += '<td style="padding: 6px; border: 1px solid var(--border); text-align: center; cursor: pointer; background: ' + bgColor + '; font-weight: bold; color: ' + (corrColors[corr] || 'var(--text-dim)') + ';" onclick="cycleCorrelation(' + Math.min(i,j) + ', ' + Math.max(i,j) + ')">' + (corrSymbols[corr] || '·') + '</td>';
                }
            });
            html += '</tr>';
        });

        html += `</table>
            <div style="margin-top: 8px; font-size: 10px; color: var(--text-dim);">
                <strong style="color:#4a9f6e">++</strong> Strong synergy &nbsp;
                <strong style="color:#7bc47f">+</strong> Synergy &nbsp;
                <strong style="color:#e74c3c">−−</strong> Strong tradeoff &nbsp;
                <strong style="color:#e8a87c">−</strong> Tradeoff
            </div>
        </div>`;
    }

    html += '</div>'; // Close flex container for correlation matrices

    container.innerHTML = html;

    // Update priority analysis
    updateQFDPriorities(priorities);
}

function cycleRelationship(whatIdx, howIdx) {
    const key = `${whatIdx}-${howIdx}`;
    const current = qfdRelationships[key] || 0;
    const cycle = [0, 1, 3, 9];
    const nextIdx = (cycle.indexOf(current) + 1) % cycle.length;
    qfdRelationships[key] = cycle[nextIdx];
    renderQFDMatrix();
}

function cycleCorrelation(how1Idx, how2Idx) {
    const key = `${Math.min(how1Idx, how2Idx)}-${Math.max(how1Idx, how2Idx)}`;
    const current = qfdCorrelations[key] || '0';
    const cycle = ['0', '+', '++', '-', '--'];
    const nextIdx = (cycle.indexOf(current) + 1) % cycle.length;
    qfdCorrelations[key] = cycle[nextIdx];
    renderQFDMatrix();
}

function cycleWhatCorrelation(what1Idx, what2Idx) {
    const key = `${what1Idx}-${what2Idx}`;
    const current = qfdWhatCorrelations[key] || '0';
    const cycle = ['0', '?', '!'];  // none, tension, conflict
    const nextIdx = (cycle.indexOf(current) + 1) % cycle.length;
    qfdWhatCorrelations[key] = cycle[nextIdx];
    renderQFDMatrix();
}

function updateQFDPriorities(priorities) {
    const sorted = [...priorities].sort((a, b) => b.score - a.score);

    document.getElementById('qfd-top-how').textContent = sorted[0]?.name || '—';

    // Coverage: what % of WHATs have at least one strong relationship?
    let coveredWhats = 0;
    qfdWhats.forEach((w, wi) => {
        const hasRelation = qfdHows.some((h, hi) => (qfdRelationships[`${wi}-${hi}`] || 0) >= 3);
        if (hasRelation) coveredWhats++;
    });
    const coverage = qfdWhats.length > 0 ? (coveredWhats / qfdWhats.length * 100) : 0;
    document.getElementById('qfd-coverage').innerHTML = `${coverage.toFixed(0)}<span class="calc-result-unit">%</span>`;

    // Count negative correlations between high-priority HOWs
    let conflicts = 0;
    Object.entries(qfdCorrelations).forEach(([key, val]) => {
        if (val === '-' || val === '--') conflicts++;
    });
    document.getElementById('qfd-conflicts').textContent = conflicts;

    // Priority bar chart
    ForgeViz.render(document.getElementById('qfd-priority-chart'), {
        title: '', chart_type: 'bar',
        traces: [{ x: priorities.map(p => p.name), y: priorities.map(p => p.score),
            name: '', trace_type: 'bar', color: '#4a9f6e' }],
        reference_lines: [], zones: [], markers: [],
        y_axis: { label: '' }, x_axis: { label: '' }
    });
}

function loadSampleQFD() {
    qfdWhats = [
        { name: 'Easy to hold', importance: 5 },
        { name: 'Long battery life', importance: 5 },
        { name: 'Fast response', importance: 4 },
        { name: 'Durable', importance: 3 },
        { name: 'Affordable', importance: 4 },
    ];

    qfdHows = [
        { name: 'Grip diameter', unit: 'mm', target: '45' },
        { name: 'Battery capacity', unit: 'mAh', target: '5000' },
        { name: 'Processor speed', unit: 'GHz', target: '2.5' },
        { name: 'Drop resistance', unit: 'm', target: '1.5' },
        { name: 'Material cost', unit: '$', target: '<15' },
    ];

    qfdRelationships = {
        '0-0': 9, '0-4': 1,
        '1-1': 9, '1-4': 3,
        '2-2': 9, '2-1': 1,
        '3-3': 9, '3-0': 1, '3-4': 3,
        '4-4': 9, '4-1': 3, '4-2': 3
    };

    qfdCorrelations = {
        '1-2': '-',  // Battery vs processor (tradeoff)
        '1-4': '--', // Battery vs cost (strong tradeoff)
        '2-4': '-',  // Processor vs cost
        '0-3': '+',  // Grip vs drop resistance
    };

    renderQFDWhats();
    renderQFDHows();
}

// Phase 2: Part Deployment
function cascadeToPhase2() {
    // Carry over HOWs as inputs to Phase 2
    const container = document.getElementById('qfd-phase2-inputs');
    container.innerHTML = qfdHows.map(h => `
        <span style="padding: 6px 12px; background: var(--accent); color: white; border-radius: 4px; font-size: 12px;">${h.name}</span>
    `).join('');

    // Initialize parts if empty
    if (qfdParts.length === 0) {
        qfdParts = [
            { name: 'Housing material' },
            { name: 'Battery cell type' },
            { name: 'Processor chip' },
        ];
    }

    setQFDPhase(2);
    renderQFDParts();
}

function renderQFDParts() {
    const container = document.getElementById('qfd-parts');
    if (!container) return;

    container.innerHTML = qfdParts.map((p, i) => `
        <div class="yamazumi-station" style="gap: 8px;">
            <input type="text" value="${p.name}" style="flex: 1; padding: 8px;"
                   oninput="qfdParts[${i}].name = this.value; renderQFDMatrix2()">
            <button class="yamazumi-station-remove" onclick="qfdParts.splice(${i}, 1); renderQFDParts()">&times;</button>
        </div>
    `).join('');

    renderQFDMatrix2();
}

function addQFDPart() {
    qfdParts.push({ name: 'New part characteristic' });
    renderQFDParts();
}

function renderQFDMatrix2() {
    const container = document.getElementById('qfd-matrix-2');
    if (!container || qfdHows.length === 0 || qfdParts.length === 0) {
        if (container) container.innerHTML = '<div style="padding: 20px; color: var(--text-dim); text-align: center;">Add part characteristics to build the matrix</div>';
        return;
    }

    const symbols = { 9: '●', 3: '○', 1: '△', 0: '' };

    let html = '<table style="border-collapse: collapse; width: 100%;">';
    html += '<tr><th style="width: 150px; padding: 8px; background: var(--bg-card); border: 1px solid var(--border);"></th>';
    qfdParts.forEach(p => {
        html += `<th style="padding: 8px; background: var(--bg-secondary); border: 1px solid var(--border); font-size: 11px;">${p.name}</th>`;
    });
    html += '</tr>';

    qfdHows.forEach((h, hi) => {
        html += `<tr><td style="padding: 8px; background: var(--bg-secondary); border: 1px solid var(--border); font-size: 12px;">${h.name}</td>`;
        qfdParts.forEach((p, pi) => {
            const key = `${hi}-${pi}`;
            const rel = qfdMatrix2[key] || 0;
            html += `<td style="padding: 4px; border: 1px solid var(--border); text-align: center; cursor: pointer; background: ${rel === 9 ? 'rgba(74, 159, 110, 0.2)' : 'transparent'};" onclick="cycleMatrix2(${hi}, ${pi})">
                <span style="font-size: 16px;">${symbols[rel]}</span>
            </td>`;
        });
        html += '</tr>';
    });

    html += '</table>';
    container.innerHTML = html;
}

function cycleMatrix2(howIdx, partIdx) {
    const key = `${howIdx}-${partIdx}`;
    const current = qfdMatrix2[key] || 0;
    const cycle = [0, 1, 3, 9];
    qfdMatrix2[key] = cycle[(cycle.indexOf(current) + 1) % cycle.length];
    renderQFDMatrix2();
}

// Phase 3: Process Planning
function cascadeToPhase3() {
    const container = document.getElementById('qfd-phase3-inputs');
    container.innerHTML = qfdParts.map(p => `
        <span style="padding: 6px 12px; background: var(--accent); color: white; border-radius: 4px; font-size: 12px;">${p.name}</span>
    `).join('');

    if (qfdProcesses.length === 0) {
        qfdProcesses = [
            { name: 'Injection pressure' },
            { name: 'Cure temperature' },
            { name: 'Assembly torque' },
        ];
    }

    setQFDPhase(3);
    renderQFDProcesses();
}

function renderQFDProcesses() {
    const container = document.getElementById('qfd-processes');
    if (!container) return;

    container.innerHTML = qfdProcesses.map((p, i) => `
        <div class="yamazumi-station" style="gap: 8px;">
            <input type="text" value="${p.name}" style="flex: 1; padding: 8px;"
                   oninput="qfdProcesses[${i}].name = this.value; renderQFDMatrix3()">
            <button class="yamazumi-station-remove" onclick="qfdProcesses.splice(${i}, 1); renderQFDProcesses()">&times;</button>
        </div>
    `).join('');

    renderQFDMatrix3();
}

function addQFDProcess() {
    qfdProcesses.push({ name: 'New process parameter' });
    renderQFDProcesses();
}

function renderQFDMatrix3() {
    const container = document.getElementById('qfd-matrix-3');
    if (!container || qfdParts.length === 0 || qfdProcesses.length === 0) {
        if (container) container.innerHTML = '<div style="padding: 20px; color: var(--text-dim); text-align: center;">Add process parameters to build the matrix</div>';
        return;
    }

    const symbols = { 9: '●', 3: '○', 1: '△', 0: '' };

    let html = '<table style="border-collapse: collapse; width: 100%;">';
    html += '<tr><th style="width: 150px; padding: 8px; background: var(--bg-card); border: 1px solid var(--border);"></th>';
    qfdProcesses.forEach(p => {
        html += `<th style="padding: 8px; background: var(--bg-secondary); border: 1px solid var(--border); font-size: 11px;">${p.name}</th>`;
    });
    html += '</tr>';

    qfdParts.forEach((part, pi) => {
        html += `<tr><td style="padding: 8px; background: var(--bg-secondary); border: 1px solid var(--border); font-size: 12px;">${part.name}</td>`;
        qfdProcesses.forEach((proc, proci) => {
            const key = `${pi}-${proci}`;
            const rel = qfdMatrix3[key] || 0;
            html += `<td style="padding: 4px; border: 1px solid var(--border); text-align: center; cursor: pointer; background: ${rel === 9 ? 'rgba(74, 159, 110, 0.2)' : 'transparent'};" onclick="cycleMatrix3(${pi}, ${proci})">
                <span style="font-size: 16px;">${symbols[rel]}</span>
            </td>`;
        });
        html += '</tr>';
    });

    html += '</table>';
    container.innerHTML = html;
}

function cycleMatrix3(partIdx, procIdx) {
    const key = `${partIdx}-${procIdx}`;
    const current = qfdMatrix3[key] || 0;
    const cycle = [0, 1, 3, 9];
    qfdMatrix3[key] = cycle[(cycle.indexOf(current) + 1) % cycle.length];
    renderQFDMatrix3();
}

// Phase 4: Production Control
function cascadeToPhase4() {
    const container = document.getElementById('qfd-phase4-inputs');
    container.innerHTML = qfdProcesses.map(p => `
        <span style="padding: 6px 12px; background: var(--accent); color: white; border-radius: 4px; font-size: 12px;">${p.name}</span>
    `).join('');

    if (qfdControls.length === 0) {
        qfdControls = [
            { name: 'Pressure gauge check' },
            { name: 'Temperature log' },
            { name: 'Torque verification' },
        ];
    }

    setQFDPhase(4);
    renderQFDControls();
}

function renderQFDControls() {
    const container = document.getElementById('qfd-controls');
    if (!container) return;

    container.innerHTML = qfdControls.map((c, i) => `
        <div class="yamazumi-station" style="gap: 8px;">
            <input type="text" value="${c.name}" style="flex: 1; padding: 8px;"
                   oninput="qfdControls[${i}].name = this.value; renderQFDMatrix4()">
            <button class="yamazumi-station-remove" onclick="qfdControls.splice(${i}, 1); renderQFDControls()">&times;</button>
        </div>
    `).join('');

    renderQFDMatrix4();
}

function addQFDControl() {
    qfdControls.push({ name: 'New control point' });
    renderQFDControls();
}

function renderQFDMatrix4() {
    const container = document.getElementById('qfd-matrix-4');
    if (!container || qfdProcesses.length === 0 || qfdControls.length === 0) {
        if (container) container.innerHTML = '<div style="padding: 20px; color: var(--text-dim); text-align: center;">Add control points to build the matrix</div>';
        return;
    }

    const symbols = { 9: '●', 3: '○', 1: '△', 0: '' };

    let html = '<table style="border-collapse: collapse; width: 100%;">';
    html += '<tr><th style="width: 150px; padding: 8px; background: var(--bg-card); border: 1px solid var(--border);"></th>';
    qfdControls.forEach(c => {
        html += `<th style="padding: 8px; background: var(--bg-secondary); border: 1px solid var(--border); font-size: 11px;">${c.name}</th>`;
    });
    html += '</tr>';

    qfdProcesses.forEach((proc, proci) => {
        html += `<tr><td style="padding: 8px; background: var(--bg-secondary); border: 1px solid var(--border); font-size: 12px;">${proc.name}</td>`;
        qfdControls.forEach((ctrl, ctrli) => {
            const key = `${proci}-${ctrli}`;
            const rel = qfdMatrix4[key] || 0;
            html += `<td style="padding: 4px; border: 1px solid var(--border); text-align: center; cursor: pointer; background: ${rel === 9 ? 'rgba(74, 159, 110, 0.2)' : 'transparent'};" onclick="cycleMatrix4(${proci}, ${ctrli})">
                <span style="font-size: 16px;">${symbols[rel]}</span>
            </td>`;
        });
        html += '</tr>';
    });

    html += '</table>';
    container.innerHTML = html;
}

function cycleMatrix4(procIdx, ctrlIdx) {
    const key = `${procIdx}-${ctrlIdx}`;
    const current = qfdMatrix4[key] || 0;
    const cycle = [0, 1, 3, 9];
    qfdMatrix4[key] = cycle[(cycle.indexOf(current) + 1) % cycle.length];
    renderQFDMatrix4();
}

function exportQFDTrace() {
    // Build traceability from controls back to customer requirements
    let trace = 'QFD Traceability Matrix\n';
    trace += '========================\n\n';

    qfdControls.forEach((ctrl, ci) => {
        trace += `Control: ${ctrl.name}\n`;

        // Find linked processes
        const linkedProcs = [];
        qfdProcesses.forEach((proc, pi) => {
            if (qfdMatrix4[`${pi}-${ci}`]) linkedProcs.push(proc.name);
        });
        trace += `  → Processes: ${linkedProcs.join(', ') || 'none'}\n`;

        // Could continue tracing back through parts to HOWs to WHATs
        trace += '\n';
    });

    // Download as text file
    const blob = new Blob([trace], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'qfd-traceability.txt';
    a.click();
    URL.revokeObjectURL(url);
}
