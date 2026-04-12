/**
 * calc-sim-flow.js — Flow & Systems Simulators for Operations Workbench
 *
 * Load order: after svend-math.js, svend-charts.js, svend-sim-core.js
 * Extracted from: calculators.html (inline script)
 *
 * Provides:
 *   Queue simulator (M/M/c live simulation)
 *   Queue A/B Compare (side-by-side scenario comparison)
 *   Cell Design simulator (U-cell, straight, L-cell layouts)
 *   Kanban Pull System simulator (push vs pull comparison)
 *   Beer Game (bullwhip effect demonstration)
 *   TOC / Drum-Buffer-Rope simulator
 */

// ============================================================================
// Queue Simulator (M/M/c Live Simulation)
// ============================================================================

let simState = {
    running: false,
    interval: null,
    queue: [],
    servers: [],
    stats: { served: 0, totalWait: 0, maxQueue: 0 },
    history: { time: [], queueLength: [] },
    simTime: 0
};

function updateSimParams() {
    const cov = parseInt(document.getElementById('qs-cov').value);
    const labels = ['0% — No variability', '15% — Low', '30% — Moderate', '50% — High', '75% — Very High', '100% — Extreme'];
    const idx = Math.min(5, Math.floor(cov / 20));
    document.getElementById('qs-cov-label').textContent = cov + '% — ' + ['No variability', 'Low', 'Moderate', 'High', 'Very High', 'Extreme'][idx];

    if (!simState.running) initSimVisual();
}

function initSimVisual() {
    const c = parseInt(document.getElementById('qs-c').value) || 3;
    simState.servers = new Array(c).fill(null);

    document.getElementById('qs-servers').innerHTML = simState.servers.map((_, i) => `
        <div id="server-${i}" style="width: 60px; height: 60px; border-radius: 50%; background: var(--bg-tertiary); border: 3px solid var(--border); display: flex; align-items: center; justify-content: center; font-size: 20px; transition: all 0.3s;">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:20px;height:20px;"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
        </div>
    `).join('');

    document.getElementById('qs-queue-visual').innerHTML = '';
    document.getElementById('qs-served').textContent = '0';
    document.getElementById('qs-current-queue').textContent = '0';
    document.getElementById('qs-avg-wait').innerHTML = '0<span class="calc-result-unit">min</span>';
    document.getElementById('qs-max-queue').textContent = '0';
}

function startSimulation() {
    if (simState.running) {
        simState.running = false;
        clearInterval(simState.interval);
        document.getElementById('qs-start').textContent = '▶ Start Simulation';
        document.getElementById('qs-start').style.background = 'var(--accent)';
        return;
    }

    simState.running = true;
    simState.queue = [];
    simState.stats = { served: 0, totalWait: 0, maxQueue: 0 };
    simState.history = { time: [], queueLength: [] };
    simState.simTime = 0;

    const c = parseInt(document.getElementById('qs-c').value) || 3;
    simState.servers = new Array(c).fill(null);

    document.getElementById('qs-start').textContent = '⏸ Pause';
    document.getElementById('qs-start').style.background = '#e74c3c';

    initSimVisual();
    simState.interval = setInterval(simStep, 100);
}

function resetSimulation() {
    simState.running = false;
    clearInterval(simState.interval);
    document.getElementById('qs-start').textContent = '▶ Start Simulation';
    document.getElementById('qs-start').style.background = 'var(--accent)';
    simState.queue = [];
    simState.stats = { served: 0, totalWait: 0, maxQueue: 0 };
    simState.history = { time: [], queueLength: [] };
    initSimVisual();
    Plotly.purge('qs-chart');
}

function simStep() {
    const speed = parseInt(document.getElementById('qs-speed').value) || 5;
    const lambda = parseFloat(document.getElementById('qs-lambda').value) || 8;
    const mu = parseFloat(document.getElementById('qs-mu').value) || 3;
    const cov = parseInt(document.getElementById('qs-cov').value) / 100;

    const dt = speed * 0.01; // hours per tick
    simState.simTime += dt;

    // Arrivals (Poisson with variability)
    const arrivalProb = lambda * dt * (1 + (Math.random() - 0.5) * 2 * cov);
    if (Math.random() < arrivalProb) {
        simState.queue.push({ arrived: simState.simTime });
    }

    // Service completions
    for (let i = 0; i < simState.servers.length; i++) {
        if (simState.servers[i]) {
            const serviceProb = mu * dt * (1 + (Math.random() - 0.5) * 2 * cov);
            if (Math.random() < serviceProb) {
                simState.stats.served++;
                simState.stats.totalWait += (simState.simTime - simState.servers[i].arrived);
                simState.servers[i] = null;
            }
        }
    }

    // Move from queue to available servers
    for (let i = 0; i < simState.servers.length; i++) {
        if (!simState.servers[i] && simState.queue.length > 0) {
            simState.servers[i] = simState.queue.shift();
        }
    }

    // Update stats
    simState.stats.maxQueue = Math.max(simState.stats.maxQueue, simState.queue.length);

    // Record history
    if (simState.history.time.length === 0 || simState.simTime - simState.history.time[simState.history.time.length - 1] > 0.01) {
        simState.history.time.push(simState.simTime);
        simState.history.queueLength.push(simState.queue.length);
        if (simState.history.time.length > 500) {
            simState.history.time.shift();
            simState.history.queueLength.shift();
        }
    }

    // Update visuals
    updateSimVisuals();
}

function updateSimVisuals() {
    // Servers
    simState.servers.forEach((s, i) => {
        const el = document.getElementById(`server-${i}`);
        if (el) {
            el.style.background = s ? 'rgba(231, 76, 60, 0.2)' : 'var(--bg-tertiary)';
            el.style.borderColor = s ? '#e74c3c' : 'var(--border)';
        }
    });

    // Queue
    const queueEl = document.getElementById('qs-queue-visual');
    const dots = simState.queue.slice(0, 30).map(() =>
        `<div style="width: 16px; height: 16px; border-radius: 50%; background: #3498db; animation: pulse 1s infinite;"></div>`
    ).join('');
    queueEl.innerHTML = dots + (simState.queue.length > 30 ? `<span style="color: var(--text-dim); font-size: 12px;">+${simState.queue.length - 30} more</span>` : '');

    // Stats
    document.getElementById('qs-served').textContent = simState.stats.served;
    document.getElementById('qs-current-queue').textContent = simState.queue.length;
    document.getElementById('qs-max-queue').textContent = simState.stats.maxQueue;
    const avgWait = simState.stats.served > 0 ? (simState.stats.totalWait / simState.stats.served * 60) : 0;
    document.getElementById('qs-avg-wait').innerHTML = `${avgWait.toFixed(1)}<span class="calc-result-unit">min</span>`;

    // Chart (update every 10 ticks)
    if (simState.history.time.length % 10 === 0) {
        Plotly.react('qs-chart', [{
            x: simState.history.time.map(t => (t * 60).toFixed(1)),
            y: simState.history.queueLength,
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            fillcolor: 'rgba(52, 152, 219, 0.2)',
            line: { color: '#3498db', width: 2 }
        }], {
            margin: { t: 10, b: 40, l: 40, r: 10 },
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#9aaa9a' },
            xaxis: { title: 'Time (min)', gridcolor: 'rgba(255,255,255,0.1)' },
            yaxis: { title: 'Queue', gridcolor: 'rgba(255,255,255,0.1)' }
        }, { responsive: true, displayModeBar: false });

        // Update burst analysis
        updateBurstAnalysis();
    }
}

// Update burst analysis in live simulator
function updateBurstAnalysis() {
    if (!simState.running || simState.stats.served < 10) return;

    const history = simState.history;
    if (history.queueLength.length < 20) return;

    // Find the peak queue moment
    const maxIdx = history.queueLength.indexOf(Math.max(...history.queueLength));
    const peakTime = history.time[maxIdx] * 60; // minutes
    const peakQueue = history.queueLength[maxIdx];

    // Calculate what a "normal" queue would be
    const avgQueue = history.queueLength.reduce((a, b) => a + b, 0) / history.queueLength.length;

    const el = document.getElementById('qs-burst-analysis');
    if (peakQueue > avgQueue * 2 && peakQueue > 3) {
        el.innerHTML = `
            <div style="color: #e74c3c; font-weight: 600; margin-bottom: 8px;">&#9888; Burst Detected</div>
            <div>Peak queue of <strong>${peakQueue}</strong> at ~${peakTime.toFixed(0)} min into simulation.</div>
            <div style="margin-top: 8px; color: var(--text-dim);">
                Average queue: ${avgQueue.toFixed(1)} | Peak was ${(peakQueue / avgQueue).toFixed(1)}x average.
                <br>This is typical arrival clustering — even with stable λ, arrivals bunch up randomly.
            </div>
        `;
    } else {
        el.innerHTML = `
            <div style="color: #27ae60;">✓ No major bursts detected</div>
            <div style="margin-top: 8px; color: var(--text-dim);">
                Max queue: ${peakQueue} | Average: ${avgQueue.toFixed(1)} — system is handling variability well.
            </div>
        `;
    }
}

// ============================================================================
// A/B Scenario Compare
// ============================================================================

let compareState = {
    running: false,
    interval: null,
    a: { queue: [], servers: [], stats: { served: 0, totalWait: 0 }, history: { time: [], queueLength: [] } },
    b: { queue: [], servers: [], stats: { served: 0, totalWait: 0 }, history: { time: [], queueLength: [] } },
    simTime: 0
};

function updateCompareParams() {
    if (!compareState.running) initCompareVisuals();
}

function initCompareVisuals() {
    const cA = parseInt(document.getElementById('qc-a-c').value) || 3;
    const cB = parseInt(document.getElementById('qc-b-c').value) || 4;

    compareState.a.servers = new Array(cA).fill(null);
    compareState.b.servers = new Array(cB).fill(null);

    document.getElementById('qc-a-servers').innerHTML = compareState.a.servers.map((_, i) =>
        `<div style="width: 40px; height: 40px; border-radius: 50%; background: var(--bg-tertiary); border: 2px solid var(--border); display: flex; align-items: center; justify-content: center; font-size: 14px;"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:20px;height:20px;"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>`
    ).join('');
    document.getElementById('qc-b-servers').innerHTML = compareState.b.servers.map((_, i) =>
        `<div style="width: 40px; height: 40px; border-radius: 50%; background: var(--bg-tertiary); border: 2px solid var(--border); display: flex; align-items: center; justify-content: center; font-size: 14px;"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:20px;height:20px;"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>`
    ).join('');

    document.getElementById('qc-a-queue').innerHTML = '';
    document.getElementById('qc-b-queue').innerHTML = '';
}

function startCompare() {
    if (compareState.running) {
        compareState.running = false;
        clearInterval(compareState.interval);
        document.getElementById('qc-start').textContent = '▶ Run Both Scenarios';
        return;
    }

    compareState.running = true;
    compareState.simTime = 0;
    compareState.a = { queue: [], servers: new Array(parseInt(document.getElementById('qc-a-c').value) || 3).fill(null), stats: { served: 0, totalWait: 0 }, history: { time: [], queueLength: [] } };
    compareState.b = { queue: [], servers: new Array(parseInt(document.getElementById('qc-b-c').value) || 4).fill(null), stats: { served: 0, totalWait: 0 }, history: { time: [], queueLength: [] } };

    document.getElementById('qc-start').textContent = '⏸ Pause';
    initCompareVisuals();
    compareState.interval = setInterval(compareStep, 80);
}

function resetCompare() {
    compareState.running = false;
    clearInterval(compareState.interval);
    document.getElementById('qc-start').textContent = '▶ Run Both Scenarios';
    document.getElementById('qc-verdict').textContent = 'Run to compare...';
    document.getElementById('qc-verdict').style.background = 'var(--bg-card)';
    initCompareVisuals();
    Plotly.purge('qc-chart');
}

function compareStep() {
    const dt = 0.05;
    compareState.simTime += dt;

    // Same random seed for both (fair comparison)
    const arrivalRoll = Math.random();
    const serviceRolls = Array(10).fill(0).map(() => Math.random());

    // Process both scenarios with same randomness
    processScenario('a', arrivalRoll, serviceRolls, dt);
    processScenario('b', arrivalRoll, serviceRolls, dt);

    updateCompareVisuals();

    // Update chart every 10 steps
    if (compareState.a.history.time.length % 10 === 0) {
        updateCompareChart();
    }
}

function processScenario(key, arrivalRoll, serviceRolls, dt) {
    const s = compareState[key];
    const lambda = parseFloat(document.getElementById(`qc-${key}-lambda`).value) || 10;
    const mu = parseFloat(document.getElementById(`qc-${key}-mu`).value) || 4;

    // Arrival
    if (arrivalRoll < lambda * dt) {
        s.queue.push({ arrived: compareState.simTime });
    }

    // Service
    for (let i = 0; i < s.servers.length; i++) {
        if (s.servers[i] && serviceRolls[i] < mu * dt) {
            s.stats.served++;
            s.stats.totalWait += (compareState.simTime - s.servers[i].arrived);
            s.servers[i] = null;
        }
    }

    // Queue to server
    for (let i = 0; i < s.servers.length; i++) {
        if (!s.servers[i] && s.queue.length > 0) {
            s.servers[i] = s.queue.shift();
        }
    }

    // History
    s.history.time.push(compareState.simTime);
    s.history.queueLength.push(s.queue.length);
    if (s.history.time.length > 300) {
        s.history.time.shift();
        s.history.queueLength.shift();
    }
}

function updateCompareVisuals() {
    ['a', 'b'].forEach(key => {
        const s = compareState[key];
        const color = key === 'a' ? '#e74c3c' : '#27ae60';

        // Servers
        const serversEl = document.getElementById(`qc-${key}-servers`);
        serversEl.innerHTML = s.servers.map(srv =>
            `<div style="width: 40px; height: 40px; border-radius: 50%; background: ${srv ? 'rgba(231,76,60,0.2)' : 'var(--bg-tertiary)'}; border: 2px solid ${srv ? '#e74c3c' : 'var(--border)'}; display: flex; align-items: center; justify-content: center; font-size: 14px;"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:20px;height:20px;"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>`
        ).join('');

        // Queue dots
        const queueEl = document.getElementById(`qc-${key}-queue`);
        queueEl.innerHTML = s.queue.slice(0, 20).map(() =>
            `<div style="width: 12px; height: 12px; border-radius: 50%; background: #3498db;"></div>`
        ).join('') + (s.queue.length > 20 ? `<span style="font-size: 10px; color: var(--text-dim);">+${s.queue.length - 20}</span>` : '');

        // Stats
        document.getElementById(`qc-${key}-served`).textContent = s.stats.served;
        const avgWait = s.stats.served > 0 ? (s.stats.totalWait / s.stats.served * 60) : 0;
        document.getElementById(`qc-${key}-wait`).textContent = avgWait.toFixed(1) + 'm';
    });

    // Verdict
    const aWait = compareState.a.stats.served > 0 ? compareState.a.stats.totalWait / compareState.a.stats.served * 60 : 0;
    const bWait = compareState.b.stats.served > 0 ? compareState.b.stats.totalWait / compareState.b.stats.served * 60 : 0;
    const verdict = document.getElementById('qc-verdict');

    if (compareState.a.stats.served > 20 && compareState.b.stats.served > 20) {
        const improvement = ((aWait - bWait) / aWait * 100);
        if (improvement > 10) {
            verdict.innerHTML = `<span style="color:#27ae60;">&#10003;</span> Scenario B reduces wait by <strong>${improvement.toFixed(0)}%</strong>`;
            verdict.style.background = 'rgba(39, 174, 96, 0.2)';
            verdict.style.color = '#27ae60';
        } else if (improvement < -10) {
            verdict.innerHTML = `<span style="color:#e74c3c;">&#9888;</span> Scenario B increases wait by <strong>${Math.abs(improvement).toFixed(0)}%</strong>`;
            verdict.style.background = 'rgba(231, 76, 60, 0.2)';
            verdict.style.color = '#e74c3c';
        } else {
            verdict.innerHTML = `&#8594; Similar performance (${Math.abs(improvement).toFixed(0)}% difference)`;
            verdict.style.background = 'rgba(241, 196, 15, 0.2)';
            verdict.style.color = '#f39c12';
        }
    }
}

function updateCompareChart() {
    Plotly.react('qc-chart', [
        {
            x: compareState.a.history.time.map(t => (t * 60).toFixed(1)),
            y: compareState.a.history.queueLength,
            type: 'scatter',
            mode: 'lines',
            name: 'Scenario A (Current)',
            line: { color: '#e74c3c', width: 2 }
        },
        {
            x: compareState.b.history.time.map(t => (t * 60).toFixed(1)),
            y: compareState.b.history.queueLength,
            type: 'scatter',
            mode: 'lines',
            name: 'Scenario B (Proposed)',
            line: { color: '#27ae60', width: 2 }
        }
    ], {
        margin: { t: 20, b: 50, l: 50, r: 20 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a' },
        xaxis: { title: 'Time (min)', gridcolor: 'rgba(255,255,255,0.1)' },
        yaxis: { title: 'Queue Length', gridcolor: 'rgba(255,255,255,0.1)' },
        legend: { orientation: 'h', y: -0.2, x: 0.5, xanchor: 'center' }
    }, { responsive: true, displayModeBar: false });
}

// ============================================================================
// Cell Design Simulator (U-cell, Straight, L-cell, Parallel layouts)
// ============================================================================

const CS_COLORS = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6'];
const CS_SCALE = 60; // 1 meter = 60 SVG units

let cellStations = [
    { name: 'S1', cycleTime: 30 },
    { name: 'S2', cycleTime: 30 },
    { name: 'S3', cycleTime: 30 },
    { name: 'S4', cycleTime: 30 },
    { name: 'S5', cycleTime: 30 },
    { name: 'S6', cycleTime: 30 },
];

let cellState = {
    running: false,
    interval: null,
    time: 0,
    layout: 'straight',
    spacing: 2.0,
    walkSpeed: 1.2,
    positions: [],
    operators: [],
    stationActive: [],
    trailSegments: new Map(),
    trailCount: 0,
    totalUnits: 0,
    history: { time: [], throughput: [], walkDist: [] },
    layoutComparison: null,
};

// --- Layout Geometry ---

function computeCellPositions(layout, n, spacingM) {
    const s = spacingM * CS_SCALE;
    const cx = 350, cy = 200;
    const pos = [];

    switch (layout) {
        case 'straight': {
            const totalW = (n - 1) * s;
            const sx = cx - totalW / 2;
            for (let i = 0; i < n; i++) pos.push({ x: sx + i * s, y: cy });
            break;
        }
        case 'u-cell': {
            const leftN = Math.ceil(n / 2);
            const rightN = n - leftN;
            const armH = (Math.max(leftN, rightN) - 1) * s;
            const uW = Math.max(s * 2, 180);
            const sy = cy - armH / 2;
            for (let i = 0; i < leftN; i++) pos.push({ x: cx - uW / 2, y: sy + i * s });
            for (let i = 0; i < rightN; i++) pos.push({ x: cx + uW / 2, y: sy + (rightN - 1 - i) * s });
            break;
        }
        case 'l-cell': {
            const hN = Math.ceil(n / 2);
            const vN = n - hN;
            const sx = cx - (hN - 1) * s / 2;
            const cornerX = sx + (hN - 1) * s;
            const cornerY = cy - vN * s / 2;
            for (let i = 0; i < hN; i++) pos.push({ x: sx + i * s, y: cornerY });
            for (let i = 0; i < vN; i++) pos.push({ x: cornerX, y: cornerY + (i + 1) * s });
            break;
        }
        case 'parallel': {
            const topN = Math.ceil(n / 2);
            const botN = n - topN;
            const rowGap = s * 1.5;
            const topY = cy - rowGap / 2;
            const botY = cy + rowGap / 2;
            const topSx = cx - (topN - 1) * s / 2;
            const botSx = cx - (botN - 1) * s / 2;
            for (let i = 0; i < topN; i++) pos.push({ x: topSx + i * s, y: topY });
            for (let i = 0; i < botN; i++) pos.push({ x: botSx + (botN - 1 - i) * s, y: botY });
            break;
        }
    }
    return pos;
}

// --- SVG Rendering ---

function renderCellDiagram() {
    const positions = computeCellPositions(cellState.layout, cellStations.length, cellState.spacing);
    cellState.positions = positions;

    // Fit viewBox to content
    if (positions.length > 0) {
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        positions.forEach(p => { minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x); minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y); });
        const pad = 80;
        const vw = Math.max(400, maxX - minX + pad * 2);
        const vh = Math.max(250, maxY - minY + pad * 2);
        const vx = minX - pad;
        const vy = minY - pad;
        document.getElementById('cs-svg').setAttribute('viewBox', `${vx} ${vy} ${vw} ${vh}`);
    }

    // Grid
    const gridG = document.getElementById('cs-svg-grid');
    let gridH = '';
    for (let gx = 0; gx < 800; gx += 60) {
        for (let gy = 0; gy < 500; gy += 60) {
            gridH += `<circle cx="${gx}" cy="${gy}" r="1" fill="rgba(255,255,255,0.06)"/>`;
        }
    }
    gridG.innerHTML = gridH;

    // Stations
    const stG = document.getElementById('cs-svg-stations');
    let stH = '';
    positions.forEach((p, i) => {
        const active = cellState.stationActive[i];
        const fill = active ? 'rgba(74,159,110,0.3)' : 'rgba(255,255,255,0.05)';
        const stroke = active ? '#4a9f6e' : 'rgba(255,255,255,0.2)';
        stH += `<g id="cs-st-${i}">`;
        stH += `<rect x="${p.x - 28}" y="${p.y - 20}" width="56" height="40" rx="5" fill="${fill}" stroke="${stroke}" stroke-width="2"/>`;
        stH += `<text x="${p.x}" y="${p.y - 4}" text-anchor="middle" fill="#ccc" font-size="12" font-weight="600">${cellStations[i].name}</text>`;
        stH += `<text x="${p.x}" y="${p.y + 12}" text-anchor="middle" fill="#888" font-size="9">${cellStations[i].cycleTime}s</text>`;
        stH += `</g>`;
    });
    stG.innerHTML = stH;

    // Operators (initial positions)
    renderCellOperators();

    // Legend
    const numOps = parseInt(document.getElementById('cs-num-operators').value) || 2;
    let legH = '';
    for (let i = 0; i < Math.min(numOps, 4); i++) {
        legH += `<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;"><span style="width:10px;height:10px;border-radius:50%;background:${CS_COLORS[i]};display:inline-block;"></span> Op ${i + 1}</div>`;
    }
    legH += `<div style="margin-top:6px;border-top:1px solid rgba(255,255,255,0.15);padding-top:6px;"><span style="color:#888;">Lines = walking trails</span></div>`;
    document.getElementById('cs-legend').innerHTML = legH;

    // Comparison chart
    updateCellComparisonChart();
}

function renderCellOperators() {
    const opG = document.getElementById('cs-svg-operators');
    let h = '';
    cellState.operators.forEach((op, i) => {
        h += `<circle id="cs-op-${i}" cx="${op.x}" cy="${op.y}" r="10" fill="${CS_COLORS[i]}" stroke="white" stroke-width="2" opacity="0.9"/>`;
        h += `<text id="cs-op-lbl-${i}" x="${op.x}" y="${op.y + 4}" text-anchor="middle" fill="white" font-size="10" font-weight="bold" pointer-events="none">${i + 1}</text>`;
    });
    opG.innerHTML = h;
}

function updateCellOperatorPos(i) {
    const op = cellState.operators[i];
    const c = document.getElementById(`cs-op-${i}`);
    const t = document.getElementById(`cs-op-lbl-${i}`);
    if (c) { c.setAttribute('cx', op.x); c.setAttribute('cy', op.y); }
    if (t) { t.setAttribute('x', op.x); t.setAttribute('y', op.y + 4); }
}

function updateCellStationHighlight(idx, active) {
    const g = document.getElementById(`cs-st-${idx}`);
    if (!g) return;
    const rect = g.querySelector('rect');
    if (rect) {
        rect.setAttribute('fill', active ? 'rgba(74,159,110,0.3)' : 'rgba(255,255,255,0.05)');
        rect.setAttribute('stroke', active ? '#4a9f6e' : 'rgba(255,255,255,0.2)');
    }
}

// --- Trail System ---

function recordCellTrail(opIdx, x1, y1, x2, y2) {
    const key = x1 < x2 || (x1 === x2 && y1 < y2)
        ? `${Math.round(x1)},${Math.round(y1)}-${Math.round(x2)},${Math.round(y2)}-${opIdx}`
        : `${Math.round(x2)},${Math.round(y2)}-${Math.round(x1)},${Math.round(y1)}-${opIdx}`;
    if (cellState.trailSegments.has(key)) {
        cellState.trailSegments.get(key).count++;
    } else {
        cellState.trailSegments.set(key, { x1, y1, x2, y2, color: CS_COLORS[opIdx], count: 1 });
    }
    cellState.trailCount++;
}

function renderCellTrails() {
    const g = document.getElementById('cs-svg-trails');
    let h = '';
    for (const [, seg] of cellState.trailSegments) {
        const opacity = Math.min(0.7, 0.08 + seg.count * 0.025);
        const width = Math.min(6, 1.5 + seg.count * 0.12);
        h += `<line x1="${seg.x1}" y1="${seg.y1}" x2="${seg.x2}" y2="${seg.y2}" stroke="${seg.color}" stroke-width="${width}" stroke-opacity="${opacity}" stroke-linecap="round"/>`;
    }
    g.innerHTML = h;
}

function clearCellTrails() {
    cellState.trailSegments = new Map();
    cellState.trailCount = 0;
    document.getElementById('cs-svg-trails').innerHTML = '';
}

// --- Operator State Machine ---

function initCellOperators() {
    const n = cellStations.length;
    const numOps = Math.min(parseInt(document.getElementById('cs-num-operators').value) || 2, 4);
    const positions = cellState.positions;

    cellState.operators = [];
    cellState.stationActive = new Array(n).fill(false);
    cellState.trailSegments = new Map();
    cellState.trailCount = 0;
    cellState.totalUnits = 0;
    cellState.time = 0;
    cellState.history = { time: [], throughput: [], walkDist: [] };

    // Split stations evenly among operators
    const perOp = Math.ceil(n / numOps);
    for (let i = 0; i < numOps; i++) {
        const start = i * perOp;
        const end = Math.min(start + perOp, n);
        const assigned = [];
        for (let j = start; j < end; j++) assigned.push(j);
        if (assigned.length === 0) continue;

        const fp = positions[assigned[0]];
        cellState.operators.push({
            x: fp.x, y: fp.y,
            assignedStations: assigned,
            currentStepIdx: 0,
            state: 'idle',
            stateTimer: 0,
            walkFrom: null, walkTo: null,
            walkProgress: 0, walkDistM: 0,
            stats: { totalWalkDist: 0, totalWalkTime: 0, totalWorkTime: 0, totalWaitTime: 0, cyclesCompleted: 0 }
        });
    }

    // Start each at their first station
    cellState.operators.forEach((op, i) => cellStartWorking(op, i));
    renderCellOperators();
    renderCellOperatorRoutes();
}

function cellStartWorking(op, opIdx) {
    const stIdx = op.assignedStations[op.currentStepIdx];
    if (cellState.stationActive[stIdx]) {
        op.state = 'waiting';
        return;
    }
    cellState.stationActive[stIdx] = true;
    updateCellStationHighlight(stIdx, true);
    op.state = 'working';
    op.stateTimer = cellStations[stIdx].cycleTime;
    op.x = cellState.positions[stIdx].x;
    op.y = cellState.positions[stIdx].y;
}

function processOperatorTick(op, opIdx, dt) {
    switch (op.state) {
        case 'idle':
            cellStartWorking(op, opIdx);
            break;

        case 'working':
            op.stateTimer -= dt;
            op.stats.totalWorkTime += dt;
            if (op.stateTimer <= 0) {
                // Done at this station
                const prevSt = op.assignedStations[op.currentStepIdx];
                cellState.stationActive[prevSt] = false;
                updateCellStationHighlight(prevSt, false);

                // Advance to next station
                op.currentStepIdx = (op.currentStepIdx + 1) % op.assignedStations.length;
                if (op.currentStepIdx === 0) {
                    op.stats.cyclesCompleted++;
                    cellState.totalUnits++;
                }

                // Calculate walk to next station
                const fromPos = cellState.positions[prevSt];
                const toSt = op.assignedStations[op.currentStepIdx];
                const toPos = cellState.positions[toSt];
                const distSVG = Math.hypot(toPos.x - fromPos.x, toPos.y - fromPos.y);
                const distM = distSVG / CS_SCALE;
                const walkTime = distM / cellState.walkSpeed;

                if (walkTime < 0.01) {
                    cellStartWorking(op, opIdx);
                } else {
                    op.state = 'walking';
                    op.walkFrom = { x: fromPos.x, y: fromPos.y };
                    op.walkTo = { x: toPos.x, y: toPos.y };
                    op.walkProgress = 0;
                    op.walkDistM = distM;
                    op.stateTimer = walkTime;
                    recordCellTrail(opIdx, fromPos.x, fromPos.y, toPos.x, toPos.y);
                }
            }
            break;

        case 'walking':
            op.stateTimer -= dt;
            op.stats.totalWalkTime += dt;
            const totalWT = op.walkDistM / cellState.walkSpeed;
            op.walkProgress = Math.min(1, 1 - (op.stateTimer / totalWT));
            op.x = op.walkFrom.x + (op.walkTo.x - op.walkFrom.x) * op.walkProgress;
            op.y = op.walkFrom.y + (op.walkTo.y - op.walkFrom.y) * op.walkProgress;
            op.stats.totalWalkDist += (dt / totalWT) * op.walkDistM;

            if (op.stateTimer <= 0) {
                op.x = op.walkTo.x;
                op.y = op.walkTo.y;
                cellStartWorking(op, opIdx);
            }
            break;

        case 'waiting':
            op.stats.totalWaitTime += dt;
            const tgt = op.assignedStations[op.currentStepIdx];
            if (!cellState.stationActive[tgt]) {
                cellStartWorking(op, opIdx);
            }
            break;
    }
}

// --- Simulation Control ---

function startCellSim() {
    if (cellState.running) {
        cellState.running = false;
        clearInterval(cellState.interval);
        document.getElementById('cs-start').innerHTML = '&#9654; Resume';
        return;
    }

    cellState.running = true;
    document.getElementById('cs-start').innerHTML = '&#9208; Pause';

    if (cellState.time === 0) {
        renderCellDiagram();
        initCellOperators();
    }

    const tick = 100;
    cellState.interval = setInterval(() => {
        const speed = parseInt(document.getElementById('cs-speed').value) || 5;
        const simDt = speed * 0.1;
        cellState.time += simDt;

        cellState.operators.forEach((op, i) => processOperatorTick(op, i, simDt));

        // Update operator dots every tick for smooth animation
        cellState.operators.forEach((op, i) => updateCellOperatorPos(i));

        // Render trails periodically
        if (Math.floor(cellState.time / 2) !== Math.floor((cellState.time - simDt) / 2)) {
            renderCellTrails();
        }

        // Update metrics every ~1 sim second
        if (Math.floor(cellState.time) !== Math.floor(cellState.time - simDt)) {
            updateCellMetrics();
            updateCellOperatorStats();
        }

        // Record history every ~5 sim seconds
        if (cellState.history.time.length === 0 || cellState.time - cellState.history.time[cellState.history.time.length - 1] >= 5) {
            const thr = cellState.time > 0 ? (cellState.totalUnits / cellState.time) * 3600 : 0;
            let wd = 0;
            cellState.operators.forEach(op => { if (op.stats.cyclesCompleted > 0) wd += op.stats.totalWalkDist / op.stats.cyclesCompleted; });
            cellState.history.time.push(cellState.time);
            cellState.history.throughput.push(thr);
            cellState.history.walkDist.push(wd);
        }
    }, tick);
}

function resetCellSim() {
    cellState.running = false;
    if (cellState.interval) clearInterval(cellState.interval);
    cellState.time = 0;
    cellState.totalUnits = 0;
    cellState.operators = [];
    cellState.stationActive = [];
    cellState.trailSegments = new Map();
    cellState.trailCount = 0;
    cellState.history = { time: [], throughput: [], walkDist: [] };
    document.getElementById('cs-start').innerHTML = '&#9654; Start Cell';
    document.getElementById('cs-svg-trails').innerHTML = '';
    document.getElementById('cs-svg-operators').innerHTML = '';
    ['cs-throughput', 'cs-walk-dist', 'cs-walk-ratio', 'cs-util', 'cs-total-dist', 'cs-cycle-time'].forEach(id => {
        document.getElementById(id).innerHTML = '&mdash;';
    });
    document.getElementById('cs-operator-stats').innerHTML = '';
    Plotly.purge('cs-compare-chart');
    renderCellDiagram();
}

// --- Metrics ---

function updateCellMetrics() {
    if (cellState.time < 0.5) return;
    let tWalkDist = 0, tWalkTime = 0, tWorkTime = 0, tWaitTime = 0, tCycles = 0;
    cellState.operators.forEach(op => {
        tWalkDist += op.stats.totalWalkDist;
        tWalkTime += op.stats.totalWalkTime;
        tWorkTime += op.stats.totalWorkTime;
        tWaitTime += op.stats.totalWaitTime;
        tCycles += op.stats.cyclesCompleted;
    });

    const thrPerHr = (cellState.totalUnits / cellState.time) * 3600;
    const avgWalkPC = tCycles > 0 ? tWalkDist / tCycles : 0;
    const walkRatio = (tWorkTime + tWalkTime) > 0 ? (tWalkTime / (tWorkTime + tWalkTime)) * 100 : 0;
    const util = (tWorkTime + tWalkTime + tWaitTime) > 0 ? (tWorkTime / (tWorkTime + tWalkTime + tWaitTime)) * 100 : 0;
    const avgCT = tCycles > 0 ? (cellState.time / cellState.totalUnits) : 0;

    document.getElementById('cs-throughput').innerHTML = `${thrPerHr.toFixed(1)}<span class="calc-result-unit">/hr</span>`;
    document.getElementById('cs-walk-dist').innerHTML = `${avgWalkPC.toFixed(1)}<span class="calc-result-unit">m</span>`;
    document.getElementById('cs-walk-ratio').innerHTML = `${walkRatio.toFixed(0)}<span class="calc-result-unit">%</span>`;
    document.getElementById('cs-util').innerHTML = `${util.toFixed(0)}<span class="calc-result-unit">%</span>`;
    document.getElementById('cs-total-dist').innerHTML = `${tWalkDist.toFixed(0)}<span class="calc-result-unit">m</span>`;
    document.getElementById('cs-cycle-time').innerHTML = `${avgCT.toFixed(1)}<span class="calc-result-unit">s</span>`;

    SvendOps.publish('cellThroughput', +thrPerHr.toFixed(1), 'units/hr', 'Cell Design');
    SvendOps.publish('cellWalkDist', +avgWalkPC.toFixed(1), 'm/cycle', 'Cell Design');
    SvendOps.publish('cellUtilization', +util.toFixed(1), '%', 'Cell Design');
}

function updateCellOperatorStats() {
    if (cellState.operators.length === 0) return;
    let h = '<table style="width:100%;border-collapse:collapse;font-size:12px;">';
    h += '<tr style="color:var(--text-dim);font-size:10px;text-transform:uppercase;letter-spacing:0.5px;">';
    h += '<th style="text-align:left;padding:6px 8px;">Operator</th>';
    h += '<th style="text-align:left;padding:6px 8px;">Stations</th>';
    h += '<th style="text-align:right;padding:6px 8px;">Walk/Cycle</th>';
    h += '<th style="text-align:right;padding:6px 8px;">Work %</th>';
    h += '<th style="text-align:right;padding:6px 8px;">Walk %</th>';
    h += '<th style="text-align:right;padding:6px 8px;">Wait %</th>';
    h += '<th style="text-align:right;padding:6px 8px;">Cycles</th>';
    h += '</tr>';
    cellState.operators.forEach((op, i) => {
        const total = op.stats.totalWorkTime + op.stats.totalWalkTime + op.stats.totalWaitTime;
        const workPct = total > 0 ? (op.stats.totalWorkTime / total * 100).toFixed(0) : 0;
        const walkPct = total > 0 ? (op.stats.totalWalkTime / total * 100).toFixed(0) : 0;
        const waitPct = total > 0 ? (op.stats.totalWaitTime / total * 100).toFixed(0) : 0;
        const walkPC = op.stats.cyclesCompleted > 0 ? (op.stats.totalWalkDist / op.stats.cyclesCompleted).toFixed(1) : '—';
        const stNames = op.assignedStations.map(s => cellStations[s].name).join(', ');
        h += `<tr style="border-top:1px solid var(--border);">`;
        h += `<td style="padding:6px 8px;"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${CS_COLORS[i]};margin-right:6px;vertical-align:middle;"></span>Op ${i + 1}</td>`;
        h += `<td style="padding:6px 8px;color:var(--text-secondary);">${stNames}</td>`;
        h += `<td style="padding:6px 8px;text-align:right;">${walkPC}m</td>`;
        h += `<td style="padding:6px 8px;text-align:right;color:#4a9f6e;">${workPct}%</td>`;
        h += `<td style="padding:6px 8px;text-align:right;color:#f39c12;">${walkPct}%</td>`;
        h += `<td style="padding:6px 8px;text-align:right;color:#888;">${waitPct}%</td>`;
        h += `<td style="padding:6px 8px;text-align:right;">${op.stats.cyclesCompleted}</td>`;
        h += `</tr>`;
    });
    h += '</table>';
    document.getElementById('cs-operator-stats').innerHTML = h;
}

// --- Layout Comparison ---

function computeCellComparison() {
    const n = cellStations.length;
    const s = parseFloat(document.getElementById('cs-spacing').value) || 2.0;
    const layouts = ['straight', 'u-cell', 'l-cell', 'parallel'];
    const results = {};
    layouts.forEach(layout => {
        const pos = computeCellPositions(layout, n, s);
        let dist = 0;
        for (let i = 0; i < n; i++) {
            const from = pos[i];
            const to = pos[(i + 1) % n];
            dist += Math.hypot(to.x - from.x, to.y - from.y) / CS_SCALE;
        }
        results[layout] = dist;
    });
    cellState.layoutComparison = results;
    return results;
}

function updateCellComparisonChart() {
    const comparison = computeCellComparison();
    const labels = ['Straight', 'U-Cell', 'L-Cell', 'Parallel'];
    const keys = ['straight', 'u-cell', 'l-cell', 'parallel'];
    const distances = keys.map(k => comparison[k]);
    const colors = keys.map(k => k === cellState.layout ? '#4a9f6e' : 'rgba(74,159,110,0.3)');

    Plotly.newPlot('cs-compare-chart', [{
        x: labels, y: distances, type: 'bar',
        marker: { color: colors },
        text: distances.map(d => d.toFixed(1) + 'm'),
        textposition: 'outside',
        textfont: { color: '#9aaa9a', size: 12 },
    }], {
        margin: { t: 20, b: 50, l: 50, r: 20 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a' },
        yaxis: { title: 'Walk Distance / Cycle (m)', gridcolor: 'rgba(255,255,255,0.06)' },
        xaxis: { gridcolor: 'rgba(255,255,255,0.06)' },
    }, { responsive: true, displayModeBar: false });

    updateCellInsights();
}

function updateCellInsights() {
    const c = cellState.layoutComparison;
    if (!c) return;
    const current = cellState.layout;
    const straightD = c['straight'];
    const currentD = c[current];
    const best = Object.entries(c).sort((a, b) => a[1] - b[1])[0];
    const layoutNames = { straight: 'Straight Line', 'u-cell': 'U-Cell', 'l-cell': 'L-Cell', parallel: 'Parallel' };
    const currentName = layoutNames[current];

    // Compare current layout vs straight baseline
    let savingsHtml;
    if (current === 'straight') {
        const bestName = layoutNames[best[0]];
        const bestSavings = straightD > 0 ? ((straightD - best[1]) / straightD * 100).toFixed(0) : 0;
        const bestSavedM = (straightD - best[1]).toFixed(1);
        savingsHtml = `
            <div style="font-size:28px;font-weight:700;color:var(--text-dim);">Baseline layout</div>
            <div style="margin-top:4px;">Straight line is the reference &mdash; other layouts are compared against it</div>
            <div style="margin-top:4px;color:var(--text-dim);">Best alternative: <strong>${bestName}</strong> saves ${bestSavings}% (${bestSavedM}m/cycle)</div>`;
    } else {
        const savings = straightD > 0 ? ((straightD - currentD) / straightD * 100).toFixed(0) : 0;
        const savedM = (straightD - currentD).toFixed(1);
        const isWorse = currentD >= straightD;
        savingsHtml = `
            <div style="font-size:28px;font-weight:700;color:${isWorse ? '#e74c3c' : '#4a9f6e'};">${isWorse ? 'No' : savings + '%'} ${isWorse ? 'improvement' : 'less walking'}</div>
            <div style="margin-top:4px;">${currentName} ${isWorse ? 'adds' : 'saves'} <strong>${Math.abs(parseFloat(savedM)).toFixed(1)}m per cycle</strong> vs straight line</div>
            <div style="margin-top:4px;color:var(--text-dim);">Best layout: <strong>${layoutNames[best[0]]}</strong> (${best[1].toFixed(1)}m/cycle)</div>`;
    }

    // Layout-specific insight bullets
    const insights = {
        straight: [
            'Straight lines are simple to set up but force long return walks',
            'Walking distance grows linearly with station count',
            'Best for single-operator, low-station-count processes',
            'Consider U-cell or L-cell when adding stations or operators'
        ],
        'u-cell': [
            'Entry and exit stations are adjacent &mdash; the return walk nearly disappears',
            'One operator can reach more stations, enabling multi-process handling',
            'Flexible staffing: add or remove operators as demand changes without reconfiguring',
            'Less walking = less fatigue, fewer errors, more value-added time'
        ],
        'l-cell': [
            'The 90&deg; turn shortens the return path vs a straight line',
            'Good when floor space constrains one dimension but not the other',
            'Operators at the corner can serve stations on both arms',
            'A compromise between straight-line simplicity and U-cell compactness'
        ],
        parallel: [
            'Two parallel rows create a loop &mdash; operators walk a circuit',
            'Keeps total footprint compact for high station counts',
            'Natural for paired processes (e.g. assembly + test on facing rows)',
            'Walking distance depends on how stations are assigned across rows'
        ]
    };

    document.getElementById('cs-insight-panel').innerHTML = `
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
            <div>
                <div style="font-weight:600;color:var(--text-primary);margin-bottom:8px;">Walking Distance: ${currentName} vs Straight</div>
                ${savingsHtml}
            </div>
            <div>
                <div style="font-weight:600;color:var(--text-primary);margin-bottom:8px;">Why ${currentName}?</div>
                <ul style="margin:0;padding-left:16px;line-height:1.8;">
                    ${(insights[current] || insights.straight).map(i => `<li>${i}</li>`).join('')}
                </ul>
            </div>
        </div>
    `;
}

// --- UI Helpers ---

function setCellLayout(layout) {
    cellState.layout = layout;
    document.querySelectorAll('.cs-layout-btn').forEach(btn => {
        const isActive = btn.dataset.layout === layout;
        btn.style.borderColor = isActive ? 'var(--accent)' : 'var(--border)';
        btn.style.background = isActive ? 'rgba(74,159,110,0.15)' : 'transparent';
        btn.style.color = isActive ? 'var(--text-primary)' : 'var(--text-secondary)';
    });
    if (cellState.running) {
        clearInterval(cellState.interval);
        cellState.running = false;
    }
    resetCellSim();
}

function updateCellParams() {
    const n = parseInt(document.getElementById('cs-num-stations').value) || 6;
    const mode = document.getElementById('cs-ct-mode').value;
    const uniformCT = parseInt(document.getElementById('cs-uniform-ct').value) || 30;

    // Show/hide sections
    document.getElementById('cs-stations-section').style.display = mode === 'custom' ? 'block' : 'none';
    document.getElementById('cs-uniform-ct-wrap').style.display = mode === 'uniform' ? 'block' : 'none';

    // Sync station count
    while (cellStations.length < n) cellStations.push({ name: `S${cellStations.length + 1}`, cycleTime: uniformCT });
    while (cellStations.length > n) cellStations.pop();

    if (mode === 'uniform') {
        cellStations.forEach(s => { s.cycleTime = uniformCT; });
    }

    cellState.spacing = parseFloat(document.getElementById('cs-spacing').value) || 2.0;
    cellState.walkSpeed = parseFloat(document.getElementById('cs-walk-speed').value) || 1.2;

    // Re-render station table if in custom mode
    if (mode === 'custom') renderCellStationTable();
    renderCellOperatorRoutes();

    if (!cellState.running) {
        renderCellDiagram();
    }
}

function renderCellStationTable() {
    let h = '<table style="width:100%;max-width:500px;border-collapse:collapse;font-size:12px;">';
    h += '<tr style="color:var(--text-dim);font-size:10px;text-transform:uppercase;"><th style="text-align:left;padding:4px 8px;">Station</th><th style="text-align:left;padding:4px 8px;">Name</th><th style="text-align:right;padding:4px 8px;">Cycle Time (s)</th></tr>';
    cellStations.forEach((s, i) => {
        h += `<tr style="border-top:1px solid var(--border);">`;
        h += `<td style="padding:6px 8px;color:var(--text-dim);">${i + 1}</td>`;
        h += `<td style="padding:6px 8px;"><input type="text" value="${s.name}" onchange="cellStations[${i}].name=this.value;renderCellDiagram();" style="background:var(--bg-secondary);border:1px solid var(--border);border-radius:4px;padding:4px 8px;color:var(--text-primary);width:80px;"></td>`;
        h += `<td style="padding:6px 8px;text-align:right;"><input type="number" value="${s.cycleTime}" min="5" max="300" onchange="cellStations[${i}].cycleTime=+this.value;renderCellDiagram();" style="background:var(--bg-secondary);border:1px solid var(--border);border-radius:4px;padding:4px 8px;color:var(--text-primary);width:70px;text-align:right;"></td>`;
        h += `</tr>`;
    });
    h += '</table>';
    document.getElementById('cs-station-table').innerHTML = h;
}

function renderCellOperatorRoutes() {
    const numOps = Math.min(parseInt(document.getElementById('cs-num-operators').value) || 2, 4);
    const n = cellStations.length;
    const perOp = Math.ceil(n / numOps);

    let h = '<div style="display:flex;flex-direction:column;gap:8px;">';
    for (let i = 0; i < numOps; i++) {
        const start = i * perOp;
        const end = Math.min(start + perOp, n);
        h += `<div style="display:flex;align-items:center;gap:8px;">`;
        h += `<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:${CS_COLORS[i]};flex-shrink:0;"></span>`;
        h += `<span style="font-size:12px;font-weight:600;color:var(--text-primary);min-width:40px;">Op ${i + 1}</span>`;
        h += `<div style="display:flex;gap:4px;flex-wrap:wrap;">`;
        for (let j = 0; j < n; j++) {
            const assigned = j >= start && j < end;
            const bg = assigned ? CS_COLORS[i] : 'transparent';
            const border = assigned ? CS_COLORS[i] : 'var(--border)';
            const color = assigned ? 'white' : 'var(--text-dim)';
            h += `<span style="display:inline-flex;align-items:center;justify-content:center;width:32px;height:24px;border-radius:4px;background:${bg};border:1px solid ${border};color:${color};font-size:10px;font-weight:600;">${cellStations[j].name}</span>`;
        }
        h += `</div></div>`;
    }
    h += '</div>';
    document.getElementById('cs-operator-routes').innerHTML = h;
}

// --- Integration ---

function importCellFromLineSim() {
    if (typeof lineStations !== 'undefined' && lineStations.length > 0) {
        cellStations = lineStations.map(s => ({ name: s.name, cycleTime: s.cycleTime }));
        document.getElementById('cs-num-stations').value = cellStations.length;
        document.getElementById('cs-ct-mode').value = 'custom';
        updateCellParams();
        resetCellSim();
    }
}

function importCellFromYamazumi() {
    if (typeof yamazumiStations !== 'undefined' && yamazumiStations.length > 0) {
        cellStations = yamazumiStations.map(s => ({ name: s.name || s.station, cycleTime: s.time || s.cycleTime || 30 }));
        document.getElementById('cs-num-stations').value = cellStations.length;
        document.getElementById('cs-ct-mode').value = 'custom';
        updateCellParams();
        resetCellSim();
    }
}

function loadVSMIntoCellSim(effectiveStations) {
    cellStations = effectiveStations.map(s => ({ name: s.name || 'Station', cycleTime: s.cycle_time || 30 }));
    document.getElementById('cs-num-stations').value = cellStations.length;
    document.getElementById('cs-ct-mode').value = 'custom';
    updateCellParams();
    resetCellSim();
    closeVSMImport();
}

// Initialize on first show (deferred — showCalc defined in inline script)
document.addEventListener('DOMContentLoaded', function initCellDesign() {
    const origShowCalc = window._origShowCalcForCell;
    if (!origShowCalc) {
        window._origShowCalcForCell = true;
        const _show = showCalc;
        showCalc = function(id) {
            _show(id);
            if (id === 'cell-sim' && cellState.positions.length === 0) {
                updateCellParams();
                renderCellDiagram();
            }
        };
    }
});

// ============================================================================
// Kanban Pull System Simulator
// ============================================================================

const kanbanStations = [
    { name: 'Stamping', cycleTime: 45 },
    { name: 'Welding', cycleTime: 55 },
    { name: 'Assembly', cycleTime: 50 },
    { name: 'Packing', cycleTime: 40 },
];

const kanbanState = {
    running: false,
    interval: null,
    mode: 'pull',
    time: 0,
    completed: 0,
    stockouts: 0,
    totalLeadTime: 0,
    stations: [],
    supermarkets: [],
    history: { time: [], wip: [] },
    customerTimer: 0,
};

function renderKanbanStations() {
    const el = document.getElementById('ks-stations');
    if (!el) return;
    el.innerHTML = kanbanStations.map((s, i) => `
        <div style="display:flex; align-items:center; gap:8px;">
            <input type="text" value="${s.name}" style="flex:1; padding:8px; background:var(--bg-secondary); border:1px solid var(--border); border-radius:6px; color:var(--text-primary);" oninput="kanbanStations[${i}].name=this.value">
            <input type="number" value="${s.cycleTime}" style="width:70px; text-align:right; padding:8px; background:var(--bg-secondary); border:1px solid var(--border); border-radius:6px; color:var(--text-primary);" oninput="kanbanStations[${i}].cycleTime=parseFloat(this.value)||30">
            <span style="color:var(--text-dim); font-size:12px; min-width:24px;">sec</span>
        </div>
    `).join('');
}

function setKanbanMode(mode) {
    kanbanState.mode = mode;
    const pushBtn = document.getElementById('ks-mode-push');
    const pullBtn = document.getElementById('ks-mode-pull');
    if (mode === 'push') {
        pushBtn.style.background = 'rgba(231,76,60,0.2)';
        pushBtn.style.color = '#e74c3c';
        pullBtn.style.background = 'var(--bg-secondary)';
        pullBtn.style.color = 'var(--text-dim)';
    } else {
        pullBtn.style.background = 'rgba(74,159,110,0.2)';
        pullBtn.style.color = 'var(--accent)';
        pushBtn.style.background = 'var(--bg-secondary)';
        pushBtn.style.color = 'var(--text-dim)';
    }
}

function updateKanbanParams() {}

function boxNormalRandom() {
    return SvendMath.randn();
}

function startKanbanSim() {
    if (kanbanState.running) {
        kanbanState.running = false;
        clearInterval(kanbanState.interval);
        document.getElementById('ks-start').innerHTML = '&#9654; Resume';
        return;
    }

    kanbanState.running = true;
    document.getElementById('ks-start').innerHTML = '&#9208; Pause';

    const nStations = kanbanStations.length;
    const kanbanCards = parseInt(document.getElementById('ks-kanban-cards').value) || 3;

    // Initialize on first run
    if (kanbanState.time === 0) {
        kanbanState.stations = kanbanStations.map(() => ({
            state: 'starved',   // working, blocked, starved
            timeRemaining: 0,
            processed: 0,
            workingTime: 0,
            blockedTime: 0,
            starvedTime: 0,
        }));
        // Supermarkets between stations (and one before station 0)
        kanbanState.supermarkets = [];
        for (let i = 0; i <= nStations; i++) {
            kanbanState.supermarkets.push({
                inventory: i === 0 ? 999 : Math.floor(kanbanCards / 2), // supplier has infinite
                kanbanCapacity: kanbanCards,
            });
        }
        kanbanState.completed = 0;
        kanbanState.stockouts = 0;
        kanbanState.totalLeadTime = 0;
        kanbanState.customerTimer = 0;
        kanbanState.history = { time: [], wip: [] };
    }

    const tickMs = 100;
    kanbanState.interval = setInterval(() => {
        const speed = parseInt(document.getElementById('ks-speed').value) || 5;
        const cov = parseInt(document.getElementById('ks-cov').value) / 100;
        const demandInterval = parseFloat(document.getElementById('ks-demand-interval').value) || 50;
        const dt = speed * 0.5; // sim seconds per tick
        const mode = kanbanState.mode;
        const kCards = parseInt(document.getElementById('ks-kanban-cards').value) || 3;

        kanbanState.time += dt;

        // Customer demand: pull from last supermarket
        kanbanState.customerTimer += dt;
        if (kanbanState.customerTimer >= demandInterval) {
            kanbanState.customerTimer -= demandInterval;
            const lastSM = kanbanState.supermarkets[nStations];
            if (lastSM.inventory > 0) {
                lastSM.inventory--;
                kanbanState.completed++;
                kanbanState.totalLeadTime += demandInterval * nStations; // simplified
            } else {
                kanbanState.stockouts++;
            }
        }

        // Process stations (forward pass)
        for (let i = 0; i < nStations; i++) {
            const stn = kanbanState.stations[i];
            const upstreamSM = kanbanState.supermarkets[i];
            const downstreamSM = kanbanState.supermarkets[i + 1];
            const baseCT = kanbanStations[i].cycleTime;

            if (stn.state === 'working') {
                stn.timeRemaining -= dt;
                stn.workingTime += dt;
                if (stn.timeRemaining <= 0) {
                    // Finished: try to place in downstream supermarket
                    if (mode === 'pull') {
                        if (downstreamSM.inventory < kCards) {
                            downstreamSM.inventory++;
                            stn.processed++;
                            stn.state = 'starved'; // need to pull new unit
                        } else {
                            stn.state = 'blocked';
                            stn.timeRemaining = 0;
                        }
                    } else {
                        // Push: always place downstream (unlimited)
                        downstreamSM.inventory++;
                        stn.processed++;
                        stn.state = 'starved';
                    }
                }
            }

            if (stn.state === 'starved') {
                // Try to pull from upstream supermarket
                if (upstreamSM.inventory > 0 || (i === 0 && mode === 'push')) {
                    if (i > 0 || mode === 'pull') upstreamSM.inventory--;
                    const actualCT = Math.max(baseCT * 0.3, baseCT * (1 + boxNormalRandom() * cov));
                    stn.timeRemaining = actualCT;
                    stn.state = 'working';
                } else {
                    stn.starvedTime += dt;
                }
            }

            if (stn.state === 'blocked') {
                stn.blockedTime += dt;
                // Retry
                if (mode === 'pull') {
                    if (downstreamSM.inventory < kCards) {
                        downstreamSM.inventory++;
                        stn.processed++;
                        stn.state = 'starved';
                    }
                } else {
                    downstreamSM.inventory++;
                    stn.processed++;
                    stn.state = 'starved';
                }
            }
        }

        // In push mode, supplier supermarket is infinite
        if (mode === 'push') {
            kanbanState.supermarkets[0].inventory = 999;
        } else {
            // Supplier replenishes when kanban returns (simplified: keep at kanban capacity)
            kanbanState.supermarkets[0].inventory = Math.min(kCards, kanbanState.supermarkets[0].inventory + 1);
        }

        // Record history
        const totalWIP = kanbanState.supermarkets.slice(1).reduce((sum, sm) => sum + sm.inventory, 0);
        if (kanbanState.history.time.length === 0 || kanbanState.time - kanbanState.history.time[kanbanState.history.time.length - 1] > 1) {
            kanbanState.history.time.push(kanbanState.time);
            kanbanState.history.wip.push(totalWIP);
            if (kanbanState.history.time.length > 500) {
                kanbanState.history.time.shift();
                kanbanState.history.wip.shift();
            }
        }

        // Update visuals
        renderKanbanVisual();
        updateKanbanMetrics();
        if (kanbanState.time > 30) updateKanbanInsights();

    }, tickMs);
}

function resetKanbanSim() {
    kanbanState.running = false;
    if (kanbanState.interval) clearInterval(kanbanState.interval);
    kanbanState.time = 0;
    kanbanState.completed = 0;
    kanbanState.stockouts = 0;
    kanbanState.totalLeadTime = 0;
    kanbanState.customerTimer = 0;
    kanbanState.stations = [];
    kanbanState.supermarkets = [];
    kanbanState.history = { time: [], wip: [] };
    document.getElementById('ks-start').innerHTML = '&#9654; Start';
    document.getElementById('ks-wip').textContent = '0';
    document.getElementById('ks-throughput').innerHTML = '0<span class="calc-result-unit">/hr</span>';
    document.getElementById('ks-leadtime').innerHTML = '0<span class="calc-result-unit">sec</span>';
    document.getElementById('ks-stockouts').textContent = '0';
    document.getElementById('ks-line').innerHTML = '<div style="color:var(--text-dim);font-size:13px;">Press Start to begin simulation</div>';
    document.getElementById('ks-insights').innerHTML = '<em>Start simulation to generate insights...</em>';
    Plotly.purge('ks-chart');
}

function renderKanbanVisual() {
    const container = document.getElementById('ks-line');
    const mode = kanbanState.mode;
    const kCards = parseInt(document.getElementById('ks-kanban-cards').value) || 3;
    let html = '';

    // Supplier
    html += '<div style="text-align:center; min-width:60px;"><div style="font-size:18px;">&#x1F69A;</div><div style="font-size:10px; color:var(--text-dim);">Supplier</div></div>';

    for (let i = 0; i < kanbanStations.length; i++) {
        const sm = kanbanState.supermarkets[i];
        const stn = kanbanState.stations[i];
        if (!stn) continue;

        // Supermarket / WIP buffer
        if (i === 0 && mode === 'push') {
            html += '<div style="text-align:center; min-width:30px; font-size:10px; color:var(--text-dim);">&#x2192;</div>';
        } else {
            const fill = mode === 'pull' ? Math.min(100, (sm.inventory / kCards) * 100) : Math.min(100, sm.inventory * 10);
            const invDisplay = mode === 'pull' ? `${sm.inventory}/${kCards}` : sm.inventory;
            html += `<div style="text-align:center; min-width:50px;">
                <div style="width:40px; height:30px; margin:0 auto; border-left:2px solid var(--border); border-right:2px solid var(--border); border-bottom:2px solid var(--border); position:relative; clip-path:polygon(20% 0%, 80% 0%, 100% 100%, 0% 100%);">
                    <div style="position:absolute; bottom:0; left:0; right:0; height:${fill}%; background:${mode === 'pull' ? 'rgba(74,159,110,0.4)' : 'rgba(231,76,60,0.4)'}; transition:height 0.2s;"></div>
                </div>
                <div style="font-size:10px; color:var(--text-dim); margin-top:2px;">${invDisplay}</div>
            </div>`;
        }

        // Station
        const stateColor = stn.state === 'working' ? '#27ae60' : stn.state === 'blocked' ? '#f39c12' : '#666';
        const stateBg = stn.state === 'working' ? 'rgba(39,174,96,0.15)' : stn.state === 'blocked' ? 'rgba(243,156,18,0.15)' : 'rgba(100,100,100,0.1)';
        html += `<div style="text-align:center; min-width:80px;">
            <div style="padding:10px 8px; border:2px solid ${stateColor}; border-radius:8px; background:${stateBg}; transition:all 0.2s;">
                <div style="font-size:12px; font-weight:600; color:var(--text-primary);">${kanbanStations[i].name}</div>
                <div style="font-size:10px; color:${stateColor}; margin-top:2px; text-transform:uppercase;">${stn.state}</div>
            </div>
        </div>`;
    }

    // Last supermarket (finished goods)
    const lastSM = kanbanState.supermarkets[kanbanStations.length];
    if (lastSM) {
        const fill = mode === 'pull' ? Math.min(100, (lastSM.inventory / kCards) * 100) : Math.min(100, lastSM.inventory * 10);
        const invDisplay = mode === 'pull' ? `${lastSM.inventory}/${kCards}` : lastSM.inventory;
        html += `<div style="text-align:center; min-width:50px;">
            <div style="width:40px; height:30px; margin:0 auto; border-left:2px solid var(--border); border-right:2px solid var(--border); border-bottom:2px solid var(--border); position:relative; clip-path:polygon(20% 0%, 80% 0%, 100% 100%, 0% 100%);">
                <div style="position:absolute; bottom:0; left:0; right:0; height:${fill}%; background:${mode === 'pull' ? 'rgba(74,159,110,0.4)' : 'rgba(231,76,60,0.4)'}; transition:height 0.2s;"></div>
            </div>
            <div style="font-size:10px; color:var(--text-dim); margin-top:2px;">${invDisplay}</div>
        </div>`;
    }

    // Customer
    html += '<div style="text-align:center; min-width:60px;"><div style="font-size:18px;">&#x1F464;</div><div style="font-size:10px; color:var(--text-dim);">Customer</div></div>';

    // Kanban return path (pull mode)
    if (mode === 'pull') {
        html += '<div style="position:absolute; bottom:-18px; left:80px; right:80px; border-bottom:1px dashed var(--accent); height:1px;"></div>';
    }

    container.style.position = 'relative';
    container.innerHTML = html;
}

function updateKanbanMetrics() {
    const totalWIP = kanbanState.supermarkets.slice(1).reduce((sum, sm) => sum + sm.inventory, 0);
    const elapsed = kanbanState.time / 3600; // hours
    const throughput = elapsed > 0 ? (kanbanState.completed / elapsed) : 0;

    document.getElementById('ks-wip').textContent = totalWIP;
    document.getElementById('ks-throughput').innerHTML = `${throughput.toFixed(1)}<span class="calc-result-unit">/hr</span>`;
    const avgLT = kanbanState.completed > 0 ? (kanbanState.totalLeadTime / kanbanState.completed) : 0;
    document.getElementById('ks-leadtime').innerHTML = `${avgLT.toFixed(0)}<span class="calc-result-unit">sec</span>`;
    document.getElementById('ks-stockouts').textContent = kanbanState.stockouts;

    // Update chart every 10 data points
    if (kanbanState.history.time.length % 5 === 0 && kanbanState.history.time.length > 0) {
        Plotly.react('ks-chart', [{
            x: kanbanState.history.time.map(t => (t / 60).toFixed(1)),
            y: kanbanState.history.wip,
            type: 'scatter', mode: 'lines', fill: 'tozeroy',
            fillcolor: kanbanState.mode === 'pull' ? 'rgba(74,159,110,0.2)' : 'rgba(231,76,60,0.2)',
            line: { color: kanbanState.mode === 'pull' ? '#4a9f6e' : '#e74c3c', width: 2 },
            name: 'WIP'
        }], {
            margin: { t: 10, b: 40, l: 40, r: 10 },
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#9aaa9a' },
            xaxis: { title: 'Time (min)', gridcolor: 'rgba(255,255,255,0.1)' },
            yaxis: { title: 'WIP (units)', gridcolor: 'rgba(255,255,255,0.1)' },
            showlegend: false
        }, { responsive: true, displayModeBar: false });
    }
}

function updateKanbanInsights() {
    const container = document.getElementById('ks-insights');
    if (!container) return;

    const mode = kanbanState.mode;
    const kCards = parseInt(document.getElementById('ks-kanban-cards').value) || 3;
    const nStations = kanbanStations.length;
    const totalWIP = kanbanState.supermarkets.slice(1).reduce((sum, sm) => sum + sm.inventory, 0);
    const elapsedHrs = kanbanState.time / 3600;
    const throughput = elapsedHrs > 0 ? (kanbanState.completed / elapsedHrs) : 0;

    // Station analysis
    let mostBlocked = { idx: -1, time: 0 };
    let mostStarved = { idx: -1, time: 0 };
    kanbanState.stations.forEach((stn, i) => {
        if (stn.blockedTime > mostBlocked.time) { mostBlocked = { idx: i, time: stn.blockedTime }; }
        if (stn.starvedTime > mostStarved.time) { mostStarved = { idx: i, time: stn.starvedTime }; }
    });

    // Find actual bottleneck (slowest station by cycle time)
    let slowestIdx = 0, slowestCT = 0;
    kanbanStations.forEach((s, i) => { if (s.cycleTime > slowestCT) { slowestCT = s.cycleTime; slowestIdx = i; } });
    const theoreticalThroughput = 3600 / slowestCT;

    // Analysis column
    let analysisHtml = '';
    analysisHtml += `<div style="margin-bottom:4px;"><strong>Mode:</strong> ${mode === 'pull' ? 'PULL (kanban-controlled)' : 'PUSH (unlimited WIP)'}</div>`;
    analysisHtml += `<div style="margin-bottom:4px;"><strong>Total WIP:</strong> ${totalWIP} units</div>`;
    analysisHtml += `<div style="margin-bottom:4px;"><strong>Throughput:</strong> ${throughput.toFixed(1)}/hr (theoretical max: ${theoreticalThroughput.toFixed(1)}/hr)</div>`;
    analysisHtml += `<div style="margin-bottom:4px;"><strong>Constraint:</strong> ${kanbanStations[slowestIdx].name} (CT: ${slowestCT}s)</div>`;
    if (kanbanState.stockouts > 0) {
        analysisHtml += `<div style="margin-bottom:4px; color:#e74c3c;"><strong>Stockouts:</strong> ${kanbanState.stockouts} — customer waited for product</div>`;
    }

    // Improvement column
    let improvHtml = '';
    if (mode === 'push') {
        improvHtml += `<div style="color:#e74c3c; margin-bottom:4px;">&#x26A0; PUSH mode: WIP is ${totalWIP} units and growing</div>`;
        improvHtml += `<div style="margin-bottom:4px;">&#x1F4A1; Switch to PULL to cap WIP at ~${kCards * nStations} units (${kCards} cards × ${nStations} loops)</div>`;
        if (totalWIP > kCards * nStations * 3) {
            improvHtml += `<div style="margin-bottom:4px;">&#x1F4C9; WIP is ${(totalWIP / (kCards * nStations)).toFixed(0)}x higher than pull-mode equivalent</div>`;
        }
    } else {
        const wipTarget = kCards * nStations;
        if (totalWIP <= wipTarget) {
            improvHtml += `<div style="color:#27ae60; margin-bottom:4px;">&#x2705; WIP controlled at ${totalWIP}/${wipTarget} max capacity</div>`;
        }
        if (kanbanState.stockouts > 0 && kanbanState.stockouts > kanbanState.completed * 0.05) {
            improvHtml += `<div style="color:#f39c12; margin-bottom:4px;">&#x26A0; Stockout rate ${((kanbanState.stockouts / (kanbanState.completed + kanbanState.stockouts)) * 100).toFixed(1)}% — consider adding kanban cards or reducing cycle times</div>`;
        } else if (kanbanState.stockouts === 0 && kanbanState.time > 120) {
            improvHtml += `<div style="color:#27ae60; margin-bottom:4px;">&#x2705; Zero stockouts — pull system is meeting demand</div>`;
        }
        if (mostBlocked.time > kanbanState.time * 0.1) {
            improvHtml += `<div style="margin-bottom:4px;">&#x1F6AB; ${kanbanStations[mostBlocked.idx].name} blocked ${((mostBlocked.time / kanbanState.time) * 100).toFixed(0)}% of the time — downstream can't keep up</div>`;
        }
        if (mostStarved.time > kanbanState.time * 0.15) {
            improvHtml += `<div style="margin-bottom:4px;">&#x23F3; ${kanbanStations[mostStarved.idx].name} starved ${((mostStarved.time / kanbanState.time) * 100).toFixed(0)}% — upstream supermarket emptying faster than replenished</div>`;
        }
    }

    // General insights
    const covVal = parseInt(document.getElementById('ks-cov').value);
    if (covVal > 30) {
        const lostPct = ((1 - throughput / theoreticalThroughput) * 100);
        if (lostPct > 10) {
            improvHtml += `<div style="margin-bottom:4px;">&#x1F4C9; High variability (${covVal}%) costing ~${lostPct.toFixed(0)}% throughput — standardize work to reduce CoV</div>`;
        }
    }

    if (!improvHtml) {
        improvHtml = `<div>&#x1F4A1; System running well. Try increasing variability or reducing kanban cards to stress-test.</div>`;
    }

    container.innerHTML = `
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
            <div>
                <div style="font-weight:600; color:var(--text-primary); margin-bottom:8px;">System Analysis</div>
                ${analysisHtml}
            </div>
            <div>
                <div style="font-weight:600; color:var(--text-primary); margin-bottom:8px;">Improvement Opportunities</div>
                ${improvHtml}
            </div>
        </div>
    `;
}

// Init kanban station inputs (deferred — DOM not ready in head)
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('ks-stations')) renderKanbanStations();
});

// ============================================================================
// Beer Game — Bullwhip Effect Simulator
// ============================================================================

const beerTiers = ['Factory', 'Distributor', 'Wholesaler', 'Retailer'];
const beerIcons = ['&#x1F3ED;', '&#x1F4E6;', '&#x1F3EA;', '&#x1F6D2;'];

const beerState = {
    running: false,
    interval: null,
    week: 0,
    tiers: [],
    history: { week: [], demand: [] },
    totalCost: 0,
};

function initBeerTiers() {
    const target = parseInt(document.getElementById('bg-target').value) || 12;
    beerState.tiers = beerTiers.map((name, i) => ({
        name,
        inventory: target,
        backlog: 0,
        pipeline: [],          // [{qty, weeksLeft}]
        lastOrder: 0,
        orderHistory: [],
        inventoryHistory: [],
    }));
    beerState.history = { week: [], demand: [] };
    beerState.totalCost = 0;
    beerState.week = 0;
}

function getBeerDemand(week) {
    const pattern = document.getElementById('bg-demand').value;
    switch (pattern) {
        case 'constant': return 4;
        case 'step': return week < 5 ? 4 : 8;
        case 'seasonal': return Math.max(1, Math.round(4 + 3 * Math.sin(week * Math.PI / 12)));
        case 'random': return Math.floor(Math.random() * 5) + 2;
        default: return 4;
    }
}

function startBeerGame() {
    if (beerState.running) {
        beerState.running = false;
        clearInterval(beerState.interval);
        document.getElementById('bg-start').innerHTML = '&#9654; Resume';
        return;
    }

    beerState.running = true;
    document.getElementById('bg-start').innerHTML = '&#9208; Pause';

    if (beerState.week === 0) initBeerTiers();

    const tickMs = 100;
    beerState.interval = setInterval(() => {
        const speed = parseInt(document.getElementById('bg-speed').value) || 4;
        // Slow down: only advance a week every N ticks
        const ticksPerWeek = Math.max(1, Math.round(10 / speed));
        if (!beerState._tickCount) beerState._tickCount = 0;
        beerState._tickCount++;
        if (beerState._tickCount < ticksPerWeek) {
            return;
        }
        beerState._tickCount = 0;

        beerState.week++;
        const leadTime = parseInt(document.getElementById('bg-leadtime').value) || 2;
        const policy = document.getElementById('bg-policy').value;
        const target = parseInt(document.getElementById('bg-target').value) || 12;
        const demand = getBeerDemand(beerState.week);

        // Record demand
        beerState.history.week.push(beerState.week);
        beerState.history.demand.push(demand);

        // Process each tier from downstream (retailer) to upstream (factory)
        for (let i = beerTiers.length - 1; i >= 0; i--) {
            const tier = beerState.tiers[i];

            // 1. Receive incoming shipments
            const arrived = [];
            const pending = [];
            for (const p of tier.pipeline) {
                p.weeksLeft--;
                if (p.weeksLeft <= 0) arrived.push(p);
                else pending.push(p);
            }
            tier.pipeline = pending;
            const received = arrived.reduce((sum, p) => sum + p.qty, 0);
            tier.inventory += received;

            // 2. Determine incoming demand
            let incomingDemand;
            if (i === beerTiers.length - 1) {
                incomingDemand = demand; // retailer faces customer demand
            } else {
                incomingDemand = beerState.tiers[i + 1].lastOrder;
            }

            // 3. Fill orders (demand + backlog)
            const totalDemand = incomingDemand + tier.backlog;
            const shipped = Math.min(tier.inventory, totalDemand);
            tier.inventory -= shipped;
            tier.backlog = totalDemand - shipped;

            // Ship downstream (into downstream tier's pipeline)
            if (i < beerTiers.length - 1) {
                // This represents what we actually ship to the tier that ordered from us
            }

            // 4. Place upstream order
            let orderQty;
            if (policy === 'match-demand') {
                orderQty = incomingDemand;
            } else {
                // Order-up-to: order enough to bring inventory position back to target
                const inventoryPosition = tier.inventory - tier.backlog + tier.pipeline.reduce((s, p) => s + p.qty, 0);
                orderQty = Math.max(0, target + incomingDemand - inventoryPosition);
            }
            tier.lastOrder = orderQty;
            tier.orderHistory.push(orderQty);
            tier.inventoryHistory.push(tier.inventory - tier.backlog);

            // Place order into upstream pipeline
            if (i > 0) {
                // Order goes to upstream tier — they'll see it as incomingDemand next week
                beerState.tiers[i - 1].pipeline.push({ qty: Math.min(orderQty, beerState.tiers[i - 1].inventory + 10), weeksLeft: leadTime });
            } else {
                // Factory: production has lead time but unlimited raw material
                tier.pipeline.push({ qty: orderQty, weeksLeft: leadTime });
            }

            // 5. Costs
            beerState.totalCost += tier.inventory * 0.5; // holding
            beerState.totalCost += tier.backlog * 1.0;   // backlog
        }

        // Update visuals
        renderBeerVisual();
        updateBeerMetrics();
        updateBeerChart();
        if (beerState.week > 3) updateBeerInsights();

    }, tickMs);
}

function resetBeerGame() {
    beerState.running = false;
    if (beerState.interval) clearInterval(beerState.interval);
    beerState.week = 0;
    beerState._tickCount = 0;
    beerState.tiers = [];
    beerState.history = { week: [], demand: [] };
    beerState.totalCost = 0;
    document.getElementById('bg-start').innerHTML = '&#9654; Start';
    document.getElementById('bg-week').textContent = '0';
    document.getElementById('bg-bullwhip').textContent = '—';
    document.getElementById('bg-cost').textContent = '$0';
    document.getElementById('bg-backlog').textContent = '0';
    document.getElementById('bg-avg-inv').textContent = '0';
    document.getElementById('bg-chain').innerHTML = '<div style="color:var(--text-dim);font-size:13px;width:100%;text-align:center;">Press Start to begin simulation</div>';
    document.getElementById('bg-insights').innerHTML = '<em>Start simulation to generate insights...</em>';
    Plotly.purge('bg-chart');
}

function updateBeerParams() {}

function renderBeerVisual() {
    const container = document.getElementById('bg-chain');
    const target = parseInt(document.getElementById('bg-target').value) || 12;
    let html = '';

    for (let i = 0; i < beerTiers.length; i++) {
        const tier = beerState.tiers[i];
        if (!tier) continue;

        const netInv = tier.inventory - tier.backlog;
        const fillPct = Math.min(100, Math.max(0, (tier.inventory / (target * 2)) * 100));
        const invColor = netInv >= 0 ? '#27ae60' : '#e74c3c';
        const invBg = netInv >= 0 ? 'rgba(39,174,96,0.15)' : 'rgba(231,76,60,0.15)';

        html += `<div style="text-align:center; flex:1; min-width:120px;">
            <div style="font-size:28px; margin-bottom:4px;">${beerIcons[i]}</div>
            <div style="font-size:12px; font-weight:600; color:var(--text-primary); margin-bottom:8px;">${tier.name}</div>
            <div style="width:60px; height:50px; margin:0 auto; background:var(--bg-tertiary); border-radius:4px; position:relative; overflow:hidden; border:1px solid var(--border);">
                <div style="position:absolute; bottom:0; left:0; right:0; height:${fillPct}%; background:${invBg}; border-top:2px solid ${invColor}; transition:height 0.3s;"></div>
            </div>
            <div style="font-size:13px; font-weight:600; color:${invColor}; margin-top:4px;">${netInv >= 0 ? netInv : netInv}</div>
            <div style="font-size:9px; color:var(--text-dim);">${netInv >= 0 ? 'inventory' : 'BACKLOG'}</div>
            <div style="font-size:10px; color:var(--text-dim); margin-top:6px; padding:3px 6px; background:var(--bg-tertiary); border-radius:4px; display:inline-block;">Order: ${tier.lastOrder}</div>
        </div>`;

        // Arrow between tiers
        if (i < beerTiers.length - 1) {
            const arrowWidth = Math.min(6, Math.max(1, tier.lastOrder / 3));
            html += `<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-width:30px; gap:4px;">
                <div style="font-size:9px; color:var(--text-dim);">ship</div>
                <div style="width:${20 + arrowWidth * 2}px; height:${arrowWidth}px; background:var(--accent); border-radius:2px; opacity:0.6;"></div>
                <div style="width:${20 + arrowWidth * 2}px; height:${Math.max(1, arrowWidth * 0.7)}px; background:#f39c12; border-radius:2px; opacity:0.4;"></div>
                <div style="font-size:9px; color:var(--text-dim);">order</div>
            </div>`;
        }
    }

    // Customer
    const demand = beerState.history.demand.length > 0 ? beerState.history.demand[beerState.history.demand.length - 1] : 0;
    html += `<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-width:30px;">
        <div style="font-size:9px; color:var(--text-dim);">demand</div>
        <div style="width:20px; height:2px; background:#3498db; border-radius:2px;"></div>
    </div>`;
    html += `<div style="text-align:center; min-width:60px;">
        <div style="font-size:28px;">&#x1F464;</div>
        <div style="font-size:12px; font-weight:600; color:var(--text-primary);">Customer</div>
        <div style="font-size:13px; color:#3498db; font-weight:600; margin-top:4px;">${demand}/wk</div>
    </div>`;

    container.innerHTML = html;
    document.getElementById('bg-week').textContent = beerState.week;
}

function updateBeerMetrics() {
    const totalBacklog = beerState.tiers.reduce((s, t) => s + t.backlog, 0);
    const totalInv = beerState.tiers.reduce((s, t) => s + t.inventory, 0);
    const avgInv = beerState.week > 0 ? (totalInv / beerTiers.length) : 0;

    document.getElementById('bg-cost').textContent = `$${beerState.totalCost.toFixed(0)}`;
    document.getElementById('bg-backlog').textContent = totalBacklog;
    document.getElementById('bg-avg-inv').textContent = avgInv.toFixed(0);

    // Bullwhip ratio: variance of factory orders / variance of customer demand
    if (beerState.history.demand.length > 5) {
        const demandArr = beerState.history.demand;
        const factoryOrders = beerState.tiers[0].orderHistory;
        const variance = (arr) => {
            const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
            return arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
        };
        const dVar = variance(demandArr);
        const fVar = variance(factoryOrders);
        const ratio = dVar > 0 ? (fVar / dVar) : 1;
        document.getElementById('bg-bullwhip').textContent = ratio.toFixed(1) + 'x';
    }
}

function updateBeerChart() {
    if (beerState.history.week.length < 2) return;
    const traces = [];
    const colors = ['#e74c3c', '#f39c12', '#9b59b6', '#27ae60'];

    for (let i = 0; i < beerTiers.length; i++) {
        if (beerState.tiers[i]) {
            traces.push({
                x: beerState.history.week,
                y: beerState.tiers[i].orderHistory,
                type: 'scatter', mode: 'lines',
                name: beerTiers[i],
                line: { color: colors[i], width: 2 },
            });
        }
    }

    // Customer demand baseline
    traces.push({
        x: beerState.history.week,
        y: beerState.history.demand,
        type: 'scatter', mode: 'lines',
        name: 'Customer Demand',
        line: { color: '#3498db', width: 2, dash: 'dot' },
    });

    Plotly.react('bg-chart', traces, {
        margin: { t: 10, b: 40, l: 40, r: 10 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a', size: 10 },
        xaxis: { title: 'Week', gridcolor: 'rgba(255,255,255,0.1)' },
        yaxis: { title: 'Units Ordered', gridcolor: 'rgba(255,255,255,0.1)' },
        legend: { orientation: 'h', y: -0.25, font: { size: 10 } },
        showlegend: true,
    }, { responsive: true, displayModeBar: false });
}

function updateBeerInsights() {
    const container = document.getElementById('bg-insights');
    if (!container || beerState.tiers.length === 0) return;

    const policy = document.getElementById('bg-policy').value;
    const leadTime = parseInt(document.getElementById('bg-leadtime').value) || 2;
    const demandPattern = document.getElementById('bg-demand').value;
    const target = parseInt(document.getElementById('bg-target').value) || 12;

    const variance = (arr) => {
        if (arr.length < 2) return 0;
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        return arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
    };

    const demandVar = variance(beerState.history.demand);
    const tierVars = beerState.tiers.map(t => variance(t.orderHistory));
    const tierBullwhip = tierVars.map(v => demandVar > 0 ? v / demandVar : 1);

    // Find worst amplifier
    let worstTier = 0, worstRatio = 0;
    tierBullwhip.forEach((r, i) => { if (r > worstRatio) { worstRatio = r; worstTier = i; } });

    const totalBacklog = beerState.tiers.reduce((s, t) => s + t.backlog, 0);
    const totalInv = beerState.tiers.reduce((s, t) => s + t.inventory, 0);
    const holdingCostPct = totalInv > 0 ? (totalInv * 0.5 / (totalInv * 0.5 + totalBacklog * 1.0) * 100) : 50;

    // Analysis
    let analysisHtml = '';
    analysisHtml += `<div style="margin-bottom:4px;"><strong>Week:</strong> ${beerState.week} &nbsp; <strong>Pattern:</strong> ${demandPattern} &nbsp; <strong>Policy:</strong> ${policy === 'match-demand' ? 'match demand' : 'order-up-to'}</div>`;
    analysisHtml += `<div style="margin-bottom:4px;"><strong>Lead time:</strong> ${leadTime} weeks per tier (${leadTime * 4} weeks end-to-end round-trip)</div>`;
    analysisHtml += `<div style="margin-bottom:4px;"><strong>Demand variance:</strong> ${demandVar.toFixed(1)}</div>`;
    for (let i = 0; i < beerTiers.length; i++) {
        const ratio = tierBullwhip[i];
        const color = ratio > 3 ? '#e74c3c' : ratio > 1.5 ? '#f39c12' : '#27ae60';
        analysisHtml += `<div style="margin-bottom:2px;"><strong>${beerTiers[i]}:</strong> order variance ${tierVars[i].toFixed(1)} → <span style="color:${color}; font-weight:600;">${ratio.toFixed(1)}x amplification</span></div>`;
    }
    analysisHtml += `<div style="margin-bottom:4px; margin-top:4px;"><strong>Total SC cost:</strong> $${beerState.totalCost.toFixed(0)} (${holdingCostPct.toFixed(0)}% holding / ${(100 - holdingCostPct).toFixed(0)}% backlog)</div>`;

    // Improvements
    let improvHtml = '';

    if (worstRatio > 3) {
        improvHtml += `<div style="color:#e74c3c; margin-bottom:4px;">&#x26A0; ${beerTiers[worstTier]} amplifying demand ${worstRatio.toFixed(1)}x — classic bullwhip</div>`;
    }

    if (policy === 'match-demand' && worstRatio > 2) {
        improvHtml += `<div style="margin-bottom:4px;">&#x1F4A1; "Match demand" policy amplifies signals — try "Order-up-to" for smoothing</div>`;
    }

    if (leadTime >= 3) {
        improvHtml += `<div style="margin-bottom:4px;">&#x1F4A1; ${leadTime}-week lead time amplifies bullwhip — reducing to ${leadTime - 1} weeks would help significantly</div>`;
    }

    if (demandPattern === 'step' && beerState.week > 8 && beerState.week < 20) {
        improvHtml += `<div style="margin-bottom:4px;">&#x1F4CA; Step change at week 5 still rippling through the chain — the system won't stabilize for ~${leadTime * 4 + 5} weeks</div>`;
    }

    if (totalBacklog > target * 2) {
        improvHtml += `<div style="color:#e74c3c; margin-bottom:4px;">&#x1F6A8; Total backlog (${totalBacklog}) exceeds ${(totalBacklog / target * 100 / beerTiers.length).toFixed(0)}% of target — service level degraded</div>`;
    }

    // Check if factory is over-ordering
    const factoryOrders = beerState.tiers[0].orderHistory;
    const recentDemand = beerState.history.demand.slice(-5);
    if (factoryOrders.length >= 5) {
        const recentOrders = factoryOrders.slice(-5);
        const avgOrder = recentOrders.reduce((a, b) => a + b, 0) / recentOrders.length;
        const avgDemand = recentDemand.reduce((a, b) => a + b, 0) / recentDemand.length;
        if (avgOrder > avgDemand * 1.8) {
            improvHtml += `<div style="margin-bottom:4px;">&#x1F3ED; Factory ordering ${avgOrder.toFixed(0)}/wk vs ${avgDemand.toFixed(0)}/wk customer demand — over-reaction building excess inventory</div>`;
        }
    }

    if (policy === 'order-up-to' && worstRatio < 1.5 && beerState.week > 15) {
        improvHtml += `<div style="color:#27ae60; margin-bottom:4px;">&#x2705; Order-up-to policy keeping bullwhip under control — ratio ${worstRatio.toFixed(1)}x</div>`;
    }

    // Countermeasures
    improvHtml += `<div style="margin-top:8px; padding-top:8px; border-top:1px solid var(--border);">`;
    improvHtml += `<div style="font-weight:600; margin-bottom:4px;">Countermeasures to try:</div>`;
    improvHtml += `<div style="margin-bottom:2px;">&#x2022; Share POS data upstream (information visibility)</div>`;
    improvHtml += `<div style="margin-bottom:2px;">&#x2022; Reduce lead time (compress the chain)</div>`;
    improvHtml += `<div style="margin-bottom:2px;">&#x2022; Smaller, more frequent orders (EPEI reduction)</div>`;
    improvHtml += `</div>`;

    container.innerHTML = `
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
            <div>
                <div style="font-weight:600; color:var(--text-primary); margin-bottom:8px;">Bullwhip Analysis</div>
                ${analysisHtml}
            </div>
            <div>
                <div style="font-weight:600; color:var(--text-primary); margin-bottom:8px;">Improvement Opportunities</div>
                ${improvHtml}
            </div>
        </div>
    `;
}

// ============================================================================
// TOC / Drum-Buffer-Rope Simulator
// ============================================================================

const tocStationsData = [
    { name: 'Station 1', capacity: 10 },
    { name: 'Station 2', capacity: 12 },
    { name: 'Station 3', capacity: 7 },
    { name: 'Station 4', capacity: 11 },
    { name: 'Station 5', capacity: 9 },
];

const tocState = {
    running: false,
    interval: null,
    mode: 'dbr',
    time: 0,
    completed: 0,
    stations: [],
    buffers: [],       // WIP between stations
    constraintIdx: -1,
    drumBuffer: 0,     // WIP in the buffer before constraint
    gateOpen: true,
    history: { time: [], wip: [], throughput: [] },
    constraintWorkingTime: 0,
};

function findConstraint() {
    let minCap = Infinity, minIdx = 0;
    for (let i = 0; i < tocStationsData.length; i++) {
        if (tocStationsData[i].capacity < minCap) {
            minCap = tocStationsData[i].capacity;
            minIdx = i;
        }
    }
    return minIdx;
}

function renderTocStations() {
    const el = document.getElementById('toc-stations');
    if (!el) return;
    const constraintIdx = findConstraint();
    el.innerHTML = tocStationsData.map((s, i) => `
        <div style="display:flex; align-items:center; gap:8px;">
            <input type="text" value="${s.name}" style="flex:1; padding:8px; background:var(--bg-secondary); border:1px solid var(--border); border-radius:6px; color:var(--text-primary);" oninput="tocStationsData[${i}].name=this.value">
            <input type="number" value="${s.capacity}" style="width:70px; text-align:right; padding:8px; background:var(--bg-secondary); border:1px solid ${i === constraintIdx ? '#e74c3c' : 'var(--border)'}; border-radius:6px; color:var(--text-primary);" oninput="tocStationsData[${i}].capacity=parseFloat(this.value)||1; renderTocStations()">
            <span style="color:var(--text-dim); font-size:12px; min-width:40px;">/hr${i === constraintIdx ? ' &#x1F941;' : ''}</span>
        </div>
    `).join('');
}

function setTocMode(mode) {
    tocState.mode = mode;
    const unBtn = document.getElementById('toc-mode-uncontrolled');
    const dbrBtn = document.getElementById('toc-mode-dbr');
    if (mode === 'uncontrolled') {
        unBtn.style.background = 'rgba(231,76,60,0.2)';
        unBtn.style.color = '#e74c3c';
        dbrBtn.style.background = 'var(--bg-secondary)';
        dbrBtn.style.color = 'var(--text-dim)';
    } else {
        dbrBtn.style.background = 'rgba(74,159,110,0.2)';
        dbrBtn.style.color = 'var(--accent)';
        unBtn.style.background = 'var(--bg-secondary)';
        unBtn.style.color = 'var(--text-dim)';
    }
}

function updateTocParams() {}

function startTocSim() {
    if (tocState.running) {
        tocState.running = false;
        clearInterval(tocState.interval);
        document.getElementById('toc-start').innerHTML = '&#9654; Resume';
        return;
    }

    tocState.running = true;
    document.getElementById('toc-start').innerHTML = '&#9208; Pause';

    const n = tocStationsData.length;
    tocState.constraintIdx = findConstraint();

    // Initialize on first run
    if (tocState.time === 0) {
        tocState.stations = tocStationsData.map((s) => ({
            state: 'starved',
            timeRemaining: 0,
            processed: 0,
            workingTime: 0,
            starvedTime: 0,
            blockedTime: 0,
        }));
        tocState.buffers = new Array(n + 1).fill(0); // buffer[0] = raw material, buffer[i] = between station i-1 and i
        tocState.buffers[0] = 50; // raw material
        tocState.drumBuffer = 0;
        tocState.completed = 0;
        tocState.constraintWorkingTime = 0;
        tocState.history = { time: [], wip: [], throughput: [] };
    }

    const tickMs = 100;
    tocState.interval = setInterval(() => {
        const speed = parseInt(document.getElementById('toc-speed').value) || 5;
        const cov = parseInt(document.getElementById('toc-cov').value) / 100;
        const bufferTarget = parseInt(document.getElementById('toc-buffer-size').value) || 5;
        const dt = speed * 0.5; // sim seconds per tick
        const mode = tocState.mode;
        const ci = tocState.constraintIdx;

        tocState.time += dt;

        // Convert capacity from units/hr to seconds/unit for cycle time
        // Process each station
        for (let i = 0; i < n; i++) {
            const stn = tocState.stations[i];
            const cap = tocStationsData[i].capacity;
            const baseCT = 3600 / cap; // seconds per unit

            if (stn.state === 'working') {
                stn.timeRemaining -= dt;
                stn.workingTime += dt;
                if (i === ci) tocState.constraintWorkingTime += dt;
                if (stn.timeRemaining <= 0) {
                    // Finished: try to place in downstream buffer
                    tocState.buffers[i + 1]++;
                    stn.processed++;
                    stn.state = 'starved'; // look for next unit
                }
            }

            if (stn.state === 'starved') {
                // Can we get material?
                let canStart = false;

                if (i === 0) {
                    // First station: check release gate
                    if (mode === 'dbr') {
                        // Rope: only release if buffer before constraint is below target
                        const bufBeforeConstraint = tocState.buffers[ci];
                        tocState.gateOpen = bufBeforeConstraint < bufferTarget;
                        if (tocState.gateOpen && tocState.buffers[0] > 0) {
                            tocState.buffers[0]--;
                            canStart = true;
                        }
                    } else {
                        // Uncontrolled: always release
                        tocState.gateOpen = true;
                        tocState.buffers[0] = 50; // infinite raw material
                        canStart = true;
                    }
                } else {
                    // Other stations: pull from upstream buffer
                    if (tocState.buffers[i] > 0) {
                        tocState.buffers[i]--;
                        canStart = true;
                    }
                }

                if (canStart) {
                    const actualCT = Math.max(baseCT * 0.3, baseCT * (1 + boxNormalRandom() * cov));
                    stn.timeRemaining = actualCT;
                    stn.state = 'working';
                } else {
                    stn.starvedTime += dt;
                }
            }
        }

        // Count output (last buffer)
        const outputBuffer = tocState.buffers[n];
        if (outputBuffer > 0) {
            tocState.completed += outputBuffer;
            tocState.buffers[n] = 0;
        }

        // In uncontrolled mode, keep raw material available
        if (mode === 'uncontrolled') {
            tocState.buffers[0] = 50;
        }

        // Record history
        const totalWIP = tocState.buffers.slice(1, n).reduce((s, v) => s + v, 0);
        const elapsedHrs = tocState.time / 3600;
        const throughput = elapsedHrs > 0 ? tocState.completed / elapsedHrs : 0;

        if (tocState.history.time.length === 0 || tocState.time - tocState.history.time[tocState.history.time.length - 1] > 1) {
            tocState.history.time.push(tocState.time);
            tocState.history.wip.push(totalWIP);
            tocState.history.throughput.push(throughput);
            if (tocState.history.time.length > 500) {
                tocState.history.time.shift();
                tocState.history.wip.shift();
                tocState.history.throughput.shift();
            }
        }

        renderTocVisual();
        updateTocMetrics();
        if (tocState.time > 30) updateTocInsights();

    }, tickMs);
}

function resetTocSim() {
    tocState.running = false;
    if (tocState.interval) clearInterval(tocState.interval);
    tocState.time = 0;
    tocState.completed = 0;
    tocState.constraintWorkingTime = 0;
    tocState.stations = [];
    tocState.buffers = [];
    tocState.history = { time: [], wip: [], throughput: [] };
    document.getElementById('toc-start').innerHTML = '&#9654; Start';
    document.getElementById('toc-throughput').innerHTML = '0<span class="calc-result-unit">/hr</span>';
    document.getElementById('toc-wip').textContent = '0';
    document.getElementById('toc-util').innerHTML = '0<span class="calc-result-unit">%</span>';
    document.getElementById('toc-leadtime').innerHTML = '0<span class="calc-result-unit">sec</span>';
    document.getElementById('toc-line').innerHTML = '<div style="color:var(--text-dim);font-size:13px;">Press Start to begin simulation</div>';
    document.getElementById('toc-insights').innerHTML = '<em>Start simulation to generate insights...</em>';
    Plotly.purge('toc-chart');
}

function renderTocVisual() {
    const container = document.getElementById('toc-line');
    const n = tocStationsData.length;
    const ci = tocState.constraintIdx;
    const mode = tocState.mode;
    const bufferTarget = parseInt(document.getElementById('toc-buffer-size').value) || 5;
    let html = '';

    // Release Gate
    const gateColor = tocState.gateOpen ? '#27ae60' : '#e74c3c';
    if (mode === 'dbr') {
        html += `<div style="text-align:center; min-width:50px;">
            <div style="width:30px; height:30px; margin:0 auto; border:2px solid ${gateColor}; border-radius:6px; display:flex; align-items:center; justify-content:center; font-size:14px; background:${tocState.gateOpen ? 'rgba(39,174,96,0.15)' : 'rgba(231,76,60,0.15)'};">
                ${tocState.gateOpen ? '&#x1F7E2;' : '&#x1F534;'}
            </div>
            <div style="font-size:9px; color:var(--text-dim); margin-top:2px;">GATE</div>
        </div>`;
    } else {
        html += `<div style="text-align:center; min-width:40px;">
            <div style="font-size:14px;">&#x27A1;</div>
            <div style="font-size:9px; color:var(--text-dim);">IN</div>
        </div>`;
    }

    for (let i = 0; i < n; i++) {
        const stn = tocState.stations[i];
        if (!stn) continue;

        // Buffer before this station (skip buffer[0] which is raw material)
        if (i > 0) {
            const bufCount = tocState.buffers[i];
            const isConstraintBuffer = (i === ci && mode === 'dbr');
            const dotCount = Math.min(10, bufCount);
            const dots = Array(dotCount).fill(0).map(() =>
                `<div style="width:8px;height:8px;border-radius:50%;background:${isConstraintBuffer ? '#f39c12' : '#3498db'};display:inline-block;margin:1px;"></div>`
            ).join('');

            html += `<div style="text-align:center; min-width:40px; max-width:60px;">
                <div style="min-height:24px; display:flex; flex-wrap:wrap; justify-content:center; align-items:center; gap:0;">${dots}</div>
                <div style="font-size:10px; color:var(--text-dim);">${bufCount}${isConstraintBuffer ? '/' + bufferTarget : ''}</div>
                ${isConstraintBuffer ? '<div style="font-size:8px; color:#f39c12; font-weight:600;">BUFFER</div>' : ''}
            </div>`;
        }

        // Station
        const isConstraint = (i === ci);
        const cap = tocStationsData[i].capacity;
        const utilPct = tocState.time > 0 ? (stn.workingTime / tocState.time * 100) : 0;
        const utilColor = utilPct > 95 ? '#e74c3c' : utilPct > 80 ? '#f39c12' : '#27ae60';
        const stateColor = stn.state === 'working' ? '#27ae60' : stn.state === 'starved' ? '#888' : '#f39c12';
        const borderColor = isConstraint ? '#e74c3c' : stateColor;
        const borderWidth = isConstraint ? '3px' : '2px';

        html += `<div style="text-align:center; min-width:80px;">
            <div style="padding:8px 6px; border:${borderWidth} solid ${borderColor}; border-radius:8px; background:${stn.state === 'working' ? 'rgba(39,174,96,0.1)' : 'rgba(100,100,100,0.05)'}; transition:all 0.2s; ${isConstraint ? 'box-shadow:0 0 8px rgba(231,76,60,0.3);' : ''}">
                ${isConstraint ? '<div style="font-size:10px; font-weight:700; color:#e74c3c; margin-bottom:2px;">&#x1F941; DRUM</div>' : ''}
                <div style="font-size:11px; font-weight:600; color:var(--text-primary);">${tocStationsData[i].name}</div>
                <div style="font-size:10px; color:var(--text-dim);">${cap}/hr</div>
                <div style="font-size:9px; color:${stateColor}; text-transform:uppercase; margin-top:2px;">${stn.state}</div>
            </div>
        </div>`;
    }

    // Output
    html += `<div style="text-align:center; min-width:50px;">
        <div style="font-size:14px;">&#x2705;</div>
        <div style="font-size:10px; color:var(--text-dim);">OUT: ${tocState.completed}</div>
    </div>`;

    // Rope visualization (DBR mode)
    if (mode === 'dbr') {
        html += `<div style="position:absolute; bottom:-20px; left:50px; right:100px; border-bottom:2px dashed var(--accent); opacity:0.4;"></div>
        <div style="position:absolute; bottom:-28px; left:50%; transform:translateX(-50%); font-size:9px; color:var(--accent); font-weight:600; letter-spacing:1px;">ROPE</div>`;
    }

    container.style.position = 'relative';
    container.innerHTML = html;
}

function updateTocMetrics() {
    const n = tocStationsData.length;
    const totalWIP = tocState.buffers.slice(1, n).reduce((s, v) => s + v, 0);
    const elapsedHrs = tocState.time / 3600;
    const throughput = elapsedHrs > 0 ? tocState.completed / elapsedHrs : 0;
    const constraintUtil = tocState.time > 0 ? (tocState.constraintWorkingTime / tocState.time * 100) : 0;

    document.getElementById('toc-wip').textContent = totalWIP;
    document.getElementById('toc-throughput').innerHTML = `${throughput.toFixed(1)}<span class="calc-result-unit">/hr</span>`;
    document.getElementById('toc-util').innerHTML = `${constraintUtil.toFixed(0)}<span class="calc-result-unit">%</span>`;

    // Simplified lead time: WIP / throughput (Little's Law)
    const lt = throughput > 0 ? (totalWIP / throughput * 3600) : 0;
    document.getElementById('toc-leadtime').innerHTML = `${lt.toFixed(0)}<span class="calc-result-unit">sec</span>`;

    // Financial throughput accounting
    const valuePerUnit = parseFloat(document.getElementById('toc-value-per-unit')?.value) || 0;
    const financialThroughput = throughput * valuePerUnit;
    const ftEl = document.getElementById('toc-financial-throughput');
    if (ftEl) ftEl.innerHTML = `$${financialThroughput.toFixed(0)}<span class="calc-result-unit">/hr</span>`;

    // Chart
    if (tocState.history.time.length % 5 === 0 && tocState.history.time.length > 0) {
        Plotly.react('toc-chart', [{
            x: tocState.history.time.map(t => (t / 60).toFixed(1)),
            y: tocState.history.wip,
            type: 'scatter', mode: 'lines', fill: 'tozeroy',
            fillcolor: tocState.mode === 'dbr' ? 'rgba(74,159,110,0.15)' : 'rgba(231,76,60,0.15)',
            line: { color: tocState.mode === 'dbr' ? '#4a9f6e' : '#e74c3c', width: 2 },
            name: 'WIP', yaxis: 'y'
        }, {
            x: tocState.history.time.map(t => (t / 60).toFixed(1)),
            y: tocState.history.throughput,
            type: 'scatter', mode: 'lines',
            line: { color: '#3498db', width: 2 },
            name: 'Throughput', yaxis: 'y2'
        }], {
            margin: { t: 10, b: 40, l: 45, r: 45 },
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#9aaa9a', size: 10 },
            xaxis: { title: 'Time (min)', gridcolor: 'rgba(255,255,255,0.1)' },
            yaxis: { title: 'WIP', gridcolor: 'rgba(255,255,255,0.1)', side: 'left' },
            yaxis2: { title: 'Throughput/hr', overlaying: 'y', side: 'right', gridcolor: 'transparent' },
            legend: { orientation: 'h', y: -0.25, font: { size: 10 } },
            showlegend: true,
        }, { responsive: true, displayModeBar: false });
    }
}

function updateTocInsights() {
    const container = document.getElementById('toc-insights');
    if (!container) return;

    const n = tocStationsData.length;
    const ci = tocState.constraintIdx;
    const mode = tocState.mode;
    const bufferTarget = parseInt(document.getElementById('toc-buffer-size').value) || 5;
    const cov = parseInt(document.getElementById('toc-cov').value);

    const totalWIP = tocState.buffers.slice(1, n).reduce((s, v) => s + v, 0);
    const elapsedHrs = tocState.time / 3600;
    const throughput = elapsedHrs > 0 ? tocState.completed / elapsedHrs : 0;
    const constraintUtil = tocState.time > 0 ? (tocState.constraintWorkingTime / tocState.time * 100) : 0;
    const constraintCap = tocStationsData[ci].capacity;
    const theoreticalMax = constraintCap; // system can never exceed constraint capacity

    // Station utilizations
    const stationUtils = tocState.stations.map((stn, i) => ({
        name: tocStationsData[i].name,
        cap: tocStationsData[i].capacity,
        util: tocState.time > 0 ? (stn.workingTime / tocState.time * 100) : 0,
        starved: tocState.time > 0 ? (stn.starvedTime / tocState.time * 100) : 0,
        isConstraint: i === ci,
    }));

    // WIP distribution
    const wipBeforeConstraint = tocState.buffers.slice(1, ci + 1).reduce((s, v) => s + v, 0);
    const wipAfterConstraint = tocState.buffers.slice(ci + 1, n).reduce((s, v) => s + v, 0);

    // Analysis
    let analysisHtml = '';
    analysisHtml += `<div style="margin-bottom:4px;"><strong>Mode:</strong> ${mode === 'dbr' ? 'Drum-Buffer-Rope' : 'Uncontrolled'}</div>`;
    analysisHtml += `<div style="margin-bottom:4px;"><strong>Constraint:</strong> ${tocStationsData[ci].name} at ${constraintCap}/hr (${mode === 'dbr' ? '&#x1F941; drum' : 'bottleneck'})</div>`;
    analysisHtml += `<div style="margin-bottom:4px;"><strong>Constraint utilization:</strong> <span style="color:${constraintUtil > 90 ? '#27ae60' : constraintUtil > 75 ? '#f39c12' : '#e74c3c'};">${constraintUtil.toFixed(0)}%</span></div>`;
    analysisHtml += `<div style="margin-bottom:4px;"><strong>System throughput:</strong> ${throughput.toFixed(1)}/hr (theoretical max: ${theoreticalMax}/hr → ${(throughput / theoreticalMax * 100).toFixed(0)}%)</div>`;
    analysisHtml += `<div style="margin-bottom:4px;"><strong>Total WIP:</strong> ${totalWIP} (${wipBeforeConstraint} before constraint, ${wipAfterConstraint} after)</div>`;

    if (mode === 'dbr') {
        const drumBuf = tocState.buffers[ci];
        analysisHtml += `<div style="margin-bottom:4px;"><strong>Drum buffer:</strong> ${drumBuf}/${bufferTarget} units</div>`;
    }

    // Station utilization breakdown
    analysisHtml += `<div style="margin-top:8px; font-weight:600; margin-bottom:4px;">Station Utilization:</div>`;
    stationUtils.forEach(s => {
        const barWidth = Math.min(100, s.util);
        const color = s.isConstraint ? '#e74c3c' : s.util > 90 ? '#f39c12' : '#27ae60';
        analysisHtml += `<div style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">
            <span style="min-width:70px; font-size:11px;">${s.name}${s.isConstraint ? ' &#x1F941;' : ''}</span>
            <div style="flex:1; height:8px; background:var(--bg-tertiary); border-radius:4px; overflow:hidden;">
                <div style="width:${barWidth}%; height:100%; background:${color}; border-radius:4px; transition:width 0.3s;"></div>
            </div>
            <span style="min-width:35px; font-size:11px; text-align:right;">${s.util.toFixed(0)}%</span>
        </div>`;
    });

    // Improvements
    let improvHtml = '';

    if (mode === 'uncontrolled') {
        improvHtml += `<div style="color:#e74c3c; margin-bottom:4px;">&#x26A0; No WIP control — inventory piling up before ${tocStationsData[ci].name}</div>`;
        if (wipBeforeConstraint > 15) {
            improvHtml += `<div style="margin-bottom:4px;">&#x1F4A1; ${wipBeforeConstraint} units queued before constraint — this is pure waste (carrying cost, floor space, lead time)</div>`;
        }
        improvHtml += `<div style="margin-bottom:4px;">&#x1F4A1; Switch to DBR to cap WIP and reduce lead time by controlling material release</div>`;

        // Check if non-constraints are being utilized when they shouldn't be
        const nonConstraintOverwork = stationUtils.filter(s => !s.isConstraint && s.util > 95);
        if (nonConstraintOverwork.length > 0) {
            improvHtml += `<div style="margin-bottom:4px;">&#x1F4CA; ${nonConstraintOverwork.map(s => s.name).join(', ')} running at >95% — producing WIP nobody downstream needs</div>`;
        }
    } else {
        // DBR mode insights
        if (constraintUtil > 92) {
            improvHtml += `<div style="color:#27ae60; margin-bottom:4px;">&#x2705; Constraint at ${constraintUtil.toFixed(0)}% utilization — buffer is protecting the drum</div>`;
        } else if (constraintUtil < 75) {
            improvHtml += `<div style="color:#f39c12; margin-bottom:4px;">&#x26A0; Constraint only at ${constraintUtil.toFixed(0)}% — buffer may be too small or upstream variability too high</div>`;
            if (bufferTarget < 5) {
                improvHtml += `<div style="margin-bottom:4px;">&#x1F4A1; Try increasing buffer from ${bufferTarget} to ${bufferTarget + 3} to better protect the drum</div>`;
            }
        }

        const drumBuf = tocState.buffers[ci];
        if (drumBuf === 0 && constraintUtil < 90) {
            improvHtml += `<div style="color:#e74c3c; margin-bottom:4px;">&#x1F6A8; Buffer empty! Constraint is starving — increase buffer size or reduce upstream variability</div>`;
        } else if (drumBuf >= bufferTarget && bufferTarget > 2) {
            improvHtml += `<div style="margin-bottom:4px;">&#x1F4A1; Buffer consistently full — you could reduce it to ${Math.max(1, bufferTarget - 2)} without starving the drum</div>`;
        }

        if (totalWIP < 10) {
            improvHtml += `<div style="color:#27ae60; margin-bottom:4px;">&#x2705; WIP controlled at ${totalWIP} units — lean flow achieved</div>`;
        }
    }

    // General Goldratt wisdom
    if (cov > 30) {
        const lostPct = ((1 - throughput / theoreticalMax) * 100);
        if (lostPct > 15) {
            improvHtml += `<div style="margin-bottom:4px;">&#x1F4C9; High variability (${cov}%) reducing throughput by ~${lostPct.toFixed(0)}%</div>`;
        }
    }

    // 5 Focusing Steps
    improvHtml += `<div style="margin-top:8px; padding-top:8px; border-top:1px solid var(--border);">`;
    improvHtml += `<div style="font-weight:600; margin-bottom:4px;">Goldratt's 5 Focusing Steps:</div>`;
    improvHtml += `<div style="margin-bottom:2px;">1. <strong>Identify</strong> the constraint → ${tocStationsData[ci].name} (${constraintCap}/hr)</div>`;
    improvHtml += `<div style="margin-bottom:2px;">2. <strong>Exploit</strong> it → maximize constraint utilization (currently ${constraintUtil.toFixed(0)}%)</div>`;
    improvHtml += `<div style="margin-bottom:2px;">3. <strong>Subordinate</strong> everything else → DBR controls non-constraint pace</div>`;
    improvHtml += `<div style="margin-bottom:2px;">4. <strong>Elevate</strong> → add capacity at constraint (increase from ${constraintCap}/hr)</div>`;
    improvHtml += `<div style="margin-bottom:2px;">5. <strong>Repeat</strong> → if constraint shifts, re-identify</div>`;
    improvHtml += `</div>`;

    container.innerHTML = `
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
            <div>
                <div style="font-weight:600; color:var(--text-primary); margin-bottom:8px;">Constraint Analysis</div>
                ${analysisHtml}
            </div>
            <div>
                <div style="font-weight:600; color:var(--text-primary); margin-bottom:8px;">Improvement Opportunities</div>
                ${improvHtml}
            </div>
        </div>
    `;
}

// Init TOC station inputs (deferred — DOM not ready in head)
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('toc-stations')) renderTocStations();
});
