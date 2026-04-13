// =============================================================================
// Mission Mode — UI Functions
// =============================================================================

let activeMission = null;
let missionStartTime = null;

function formatSimTime(seconds) {
    return SvendMath.formatSimTime(seconds);
}

// --- Alert Bar ---
function renderAlertBar() {
    const bar = document.getElementById('alert-bar');
    if (!des || !des.missionMode || des.missionAlerts.length === 0) {
        bar.classList.remove('active');
        return;
    }
    bar.classList.add('active');
    const active = des.missionAlerts.filter(a => !a.acknowledged).slice(-5).reverse();
    const acked = des.missionAlerts.filter(a => a.acknowledged).slice(-2).reverse();
    bar.innerHTML = [...active, ...acked].map(a => {
        const timeStr = formatSimTime(a.time - des.warmupTime);
        return `<div class="alert-item ${a.acknowledged ? 'acknowledged' : ''} severity-${a.severity}"
                     onclick="acknowledgeAlert(${a.id})">
            <span class="alert-severity-dot ${a.severity}"></span>
            <span class="alert-time">${timeStr}</span>
            <span class="alert-message">${a.message}</span>
            ${!a.acknowledged ? `<button class="alert-ack-btn" onclick="event.stopPropagation();acknowledgeAlert(${a.id})">ACK</button>` : ''}
        </div>`;
    }).join('');
}

function acknowledgeAlert(alertId) {
    if (des) des.interventionAcknowledgeAlert(alertId);
    renderAlertBar();
    updateCommandPanel();
}

// --- Command Panel ---
function showCommandPanel() {
    document.getElementById('command-panel').classList.add('active');
    hideTeachingPanel();
}
function hideCommandPanel() {
    document.getElementById('command-panel').classList.remove('active');
    document.getElementById('command-restore-btn').style.display = 'none';
}
function collapseCommandPanel(collapse) {
    document.getElementById('command-panel').classList.toggle('active', !collapse);
    document.getElementById('command-restore-btn').style.display = collapse ? '' : 'none';
}

function updateCommandPanel() {
    if (!des || !des.missionMode) return;
    const elapsed = Math.max(0, des.clock - des.warmupTime);
    document.getElementById('cmd-clock').textContent = formatSimTime(elapsed);

    if (activeMission && activeMission.briefing) {
        document.getElementById('cmd-situation').textContent = activeMission.briefing;
    }

    const threats = des.missionAlerts.filter(a => !a.acknowledged);
    const threatEl = document.getElementById('cmd-threats-list');
    if (threats.length === 0) {
        threatEl.innerHTML = '<div style="color:var(--text-dim);font-size:0.6rem;">No active threats</div>';
    } else {
        threatEl.innerHTML = threats.map(t =>
            `<div style="padding:3px 0;font-size:0.65rem;color:${t.severity === 'critical' ? '#fca5a5' : t.severity === 'warning' ? '#fcd34d' : '#93c5fd'};">
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;color:var(--text-dim);">${formatSimTime(t.time - des.warmupTime)}</span>
                ${t.message}
            </div>`
        ).join('');
    }

    renderCommandActions();
    renderDecisionLog();
}

function renderCommandActions() {
    const el = document.getElementById('cmd-actions-list');
    if (!des) { el.innerHTML = ''; return; }
    let html = '';

    for (const [id, stn] of des.stations) {
        if (stn.type !== 'machine') continue;
        if (stn.state === 'down' && !stn._manualStop) {
            html += `<button class="command-btn" onclick="cmdForceRepair('${id}')">Repair: ${stn.name}</button>`;
            if (des.maintenanceCrew.size > 0) {
                html += `<button class="command-btn" onclick="cmdMaintPriority('${id}')">Prioritize Maint: ${stn.name}</button>`;
            }
        }
        if (stn._manualStop) {
            html += `<button class="command-btn" onclick="cmdStartMachine('${id}')">Restart: ${stn.name}</button>`;
        } else if (stn.state !== 'down') {
            html += `<button class="command-btn destructive" onclick="cmdStopMachine('${id}')">Stop: ${stn.name}</button>`;
        }
    }

    if (des.operators.filter(o => o.status !== 'quit').length > 0) {
        html += `<button class="command-btn" onclick="showReassignDialog()">Reassign Operator</button>`;
    }

    html += `<button class="command-btn" onclick="cmdToggleOT()">${des.overtimeActive ? 'Cancel Overtime' : 'Authorize Overtime'}</button>`;
    html += `<button class="command-btn" onclick="showDispatchDialog()">Change Dispatch Rule</button>`;

    const stnsWithWIP = [...des.stations.values()].filter(s => s.type === 'machine' && s.queue.length > 0);
    if (stnsWithWIP.length > 0) {
        html += `<button class="command-btn destructive" onclick="showScrapDialog()">Scrap Suspect WIP</button>`;
    }

    if (des.paused) {
        html += `<button class="command-btn resume-btn" onclick="resumeMission()">RESUME SIMULATION</button>`;
    }

    el.innerHTML = html;
}

function renderDecisionLog() {
    const el = document.getElementById('cmd-decision-log');
    if (!des || des.decisionLog.length === 0) {
        el.innerHTML = '<div style="font-size:0.6rem;color:var(--text-dim);">No decisions recorded yet.</div>';
        return;
    }
    el.innerHTML = des.decisionLog.slice().reverse().map(d =>
        `<div class="decision-entry">
            <span class="de-time">${formatSimTime(d.time - des.warmupTime)}</span>
            ${d.detail}
        </div>`
    ).join('');
}

// --- Command Action Handlers ---
function cmdForceRepair(stnId) { if (des) { des.interventionForceRepair(stnId); updateCommandPanel(); } }
function cmdMaintPriority(stnId) { if (des) { des.interventionRequestMaintPriority(stnId); updateCommandPanel(); } }
function cmdStopMachine(stnId) { if (des) { des.interventionStopMachine(stnId); updateCommandPanel(); renderCanvas(); } }
function cmdStartMachine(stnId) { if (des) { des.interventionStartMachine(stnId); updateCommandPanel(); renderCanvas(); } }
function cmdToggleOT() { if (des) { des.interventionAuthorizeOT(!des.overtimeActive); updateCommandPanel(); } }

function resumeMission() {
    if (!des) return;
    des.paused = false;
    const speed = parseFloat(document.getElementById('cfg-speed').value) || 20;
    des.running = true;
    const startReal = performance.now();
    const startClock = des.clock;
    const endTime = des.warmupTime + des.runTime;
    const tick = () => {
        if (!des || !des.running) return;
        if (des.paused) {
            updateLiveMetrics(des.getState());
            renderCanvas();
            updateCommandPanel();
            renderAlertBar();
            animFrameId = requestAnimationFrame(tick);
            return;
        }
        const realElapsed = (performance.now() - startReal) / 1000;
        const targetTime = startClock + realElapsed * speed;
        let eventsProcessed = 0;
        while (des.eventQueue.size > 0 && des.eventQueue.peek().time <= targetTime && eventsProcessed < 500) {
            if (!des.processNextEvent()) { des.running = false; break; }
            eventsProcessed++;
            if (des.paused) break;
        }
        updateLiveMetrics(des.getState());
        renderCanvas();
        updateCommandPanel();
        renderAlertBar();
        if (des.running && des.clock < endTime) {
            animFrameId = requestAnimationFrame(tick);
        } else {
            des.running = false;
            updateLiveMetrics(des.getState());
            document.getElementById('btn-play').innerHTML = '&#9654; Play';
            const results = des.getResults();
            showResults(results);
            showAfterActionReview(results);
        }
    };
    document.getElementById('btn-play').innerHTML = '&#9646;&#9646; Pause';
    animFrameId = requestAnimationFrame(tick);
    updateCommandPanel();
}

function showReassignDialog() {
    if (!des) return;
    const ops = des.operators.filter(o => o.status !== 'quit');
    const stns = [...des.stations.values()].filter(s => s.type === 'machine');
    const panel = document.getElementById('props-panel');
    const content = document.getElementById('props-content');
    panel.classList.add('active');
    content.innerHTML = `
        <div style="font-size:0.8rem;font-weight:600;margin-bottom:6px;">Reassign Operator</div>
        <div class="prop-row"><span class="prop-label">Operator</span>
            <select class="prop-input" id="reassign-op">
                ${ops.map(o => `<option value="${o.id}">${o.name} [${o.status}]</option>`).join('')}
            </select>
        </div>
        <div class="prop-row"><span class="prop-label">Station</span>
            <select class="prop-input" id="reassign-stn">
                ${stns.map(s => `<option value="${s.id}">${s.name} [q:${s.queue.length}]</option>`).join('')}
            </select>
        </div>
        <button class="sim-btn" onclick="executeReassign()" style="width:100%;margin-top:6px;">Reassign</button>
    `;
}
function executeReassign() {
    const opId = document.getElementById('reassign-op')?.value;
    const stnId = document.getElementById('reassign-stn')?.value;
    if (opId && stnId && des) {
        des.interventionReassignOperator(opId, stnId);
        updateCommandPanel();
        showToast('Operator reassigned');
        document.getElementById('props-panel').classList.remove('active');
    }
}

function showDispatchDialog() {
    if (!des) return;
    const stns = [...des.stations.values()].filter(s => s.type === 'machine');
    const panel = document.getElementById('props-panel');
    const content = document.getElementById('props-content');
    panel.classList.add('active');
    content.innerHTML = `
        <div style="font-size:0.8rem;font-weight:600;margin-bottom:6px;">Change Dispatch Rule</div>
        <div class="prop-row"><span class="prop-label">Station</span>
            <select class="prop-input" id="dispatch-stn">
                ${stns.map(s => `<option value="${s.id}">${s.name} [${s.dispatchRule || 'FIFO'}]</option>`).join('')}
            </select>
        </div>
        <div class="prop-row"><span class="prop-label">Rule</span>
            <select class="prop-input" id="dispatch-rule">
                ${['FIFO','SPT','EDD','CR','WSPT'].map(r => `<option value="${r}">${r}</option>`).join('')}
            </select>
        </div>
        <button class="sim-btn" onclick="executeDispatchChange()" style="width:100%;margin-top:6px;">Set Rule</button>
    `;
}
function executeDispatchChange() {
    const stnId = document.getElementById('dispatch-stn')?.value;
    const rule = document.getElementById('dispatch-rule')?.value;
    if (stnId && rule && des) {
        des.interventionChangeDispatchRule(stnId, rule);
        updateCommandPanel();
        showToast('Dispatch rule changed');
        document.getElementById('props-panel').classList.remove('active');
    }
}

function showScrapDialog() {
    if (!des) return;
    const stns = [...des.stations.values()].filter(s => s.type === 'machine' && s.queue.length > 0);
    if (stns.length === 0) { showToast('No queued WIP to scrap'); return; }
    const panel = document.getElementById('props-panel');
    const content = document.getElementById('props-content');
    panel.classList.add('active');
    content.innerHTML = `
        <div style="font-size:0.8rem;font-weight:600;margin-bottom:6px;">Scrap Suspect WIP</div>
        <div class="prop-row"><span class="prop-label">Station</span>
            <select class="prop-input" id="scrap-stn">
                ${stns.map(s => `<option value="${s.id}">${s.name} [${s.queue.length} queued]</option>`).join('')}
            </select>
        </div>
        <button class="sim-btn" onclick="executeScrap()" style="width:100%;margin-top:6px;background:#e74c3c;color:#fff;">Scrap WIP</button>
    `;
}
function executeScrap() {
    const stnId = document.getElementById('scrap-stn')?.value;
    if (stnId && des) {
        des.interventionScrapSuspectWIP(stnId);
        updateCommandPanel();
        renderCanvas();
        showToast('WIP scrapped');
        document.getElementById('props-panel').classList.remove('active');
    }
}

// =============================================================================
// Live Mission Dashboard — Real-time Charts, SPC, Evidence
// =============================================================================

let liveChartsInitialized = false;
let liveChartsReady = false;  // set after Plotly.newPlot completes
let liveUpdateCounter = 0;
let liveThroughputObs = [];  // observation buffer for SPC
let liveYieldObs = [];

const PLOTLY_LIVE_LAYOUT = SvendCharts.LIVE_LAYOUT;

function initLiveCharts() {
    if (liveChartsInitialized) return;
    liveChartsInitialized = true;
    liveThroughputObs = [];
    liveYieldObs = [];
    liveUpdateCounter = 0;

    // Defer Plotly init to next frame so containers are visible and have dimensions
    requestAnimationFrame(() => {
        const opts = { responsive: true, displayModeBar: false };

        Plotly.newPlot('live-chart-throughput', [
            { y: [], type: 'scatter', mode: 'lines', line: { color: '#4a9f6e', width: 1.5 }, name: 'Throughput' },
        ], { ...PLOTLY_LIVE_LAYOUT, title: { text: 'Throughput (units/hr)', font: { size: 10, color: '#9aaa9a' } } }, opts);

        Plotly.newPlot('live-chart-wip', [
            { y: [], type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 1.5 }, name: 'WIP' },
        ], { ...PLOTLY_LIVE_LAYOUT, title: { text: 'WIP', font: { size: 10, color: '#9aaa9a' } } }, opts);

        Plotly.newPlot('live-chart-yield', [
            { y: [], type: 'scatter', mode: 'lines', line: { color: '#f59e0b', width: 1.5 }, name: 'Yield' },
        ], { ...PLOTLY_LIVE_LAYOUT, title: { text: 'Yield %', font: { size: 10, color: '#9aaa9a' } }, yaxis: { ...PLOTLY_LIVE_LAYOUT.yaxis, range: [0, 105] } }, opts);

        Plotly.newPlot('live-chart-cost', [
            { y: [], type: 'scatter', mode: 'lines', line: { color: '#e74c3c', width: 1.5 }, name: 'Cost' },
        ], { ...PLOTLY_LIVE_LAYOUT, title: { text: 'Total Cost ($)', font: { size: 10, color: '#9aaa9a' } } }, opts);

        // SPC charts — throughput and yield with control limits
        Plotly.newPlot('spc-chart-throughput', [
            { y: [], type: 'scatter', mode: 'lines+markers', line: { color: '#4a9f6e', width: 1 }, marker: { size: 3 }, name: 'Throughput' },
            { y: [], type: 'scatter', mode: 'lines', line: { color: '#e74c3c', width: 1, dash: 'dash' }, name: 'UCL' },
            { y: [], type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 1, dash: 'dot' }, name: 'CL' },
            { y: [], type: 'scatter', mode: 'lines', line: { color: '#e74c3c', width: 1, dash: 'dash' }, name: 'LCL' },
            { y: [], x: [], type: 'scatter', mode: 'markers', marker: { size: 7, color: '#e74c3c', symbol: 'x' }, name: 'Signal' },
        ], { ...PLOTLY_LIVE_LAYOUT, title: { text: 'I-Chart: Throughput Rate', font: { size: 10, color: '#9aaa9a' } }, showlegend: true }, opts);

        Plotly.newPlot('spc-chart-yield', [
            { y: [], type: 'scatter', mode: 'lines+markers', line: { color: '#f59e0b', width: 1 }, marker: { size: 3 }, name: 'Yield' },
            { y: [], type: 'scatter', mode: 'lines', line: { color: '#e74c3c', width: 1, dash: 'dash' }, name: 'UCL' },
            { y: [], type: 'scatter', mode: 'lines', line: { color: '#3b82f6', width: 1, dash: 'dot' }, name: 'CL' },
            { y: [], type: 'scatter', mode: 'lines', line: { color: '#e74c3c', width: 1, dash: 'dash' }, name: 'LCL' },
            { y: [], x: [], type: 'scatter', mode: 'markers', marker: { size: 7, color: '#e74c3c', symbol: 'x' }, name: 'Signal' },
        ], { ...PLOTLY_LIVE_LAYOUT, title: { text: 'I-Chart: Yield %', font: { size: 10, color: '#9aaa9a' } }, showlegend: true, yaxis: { ...PLOTLY_LIVE_LAYOUT.yaxis, range: [0, 105] } }, opts);

        liveChartsReady = true;
    });
}

function updateLiveCharts(state) {
    if (!liveChartsReady || !state) return;
    liveUpdateCounter++;
    // Throttle Plotly updates to every 5th frame
    if (liveUpdateCounter % 5 !== 0) return;

    const tp = state.throughput || 0;
    const wip = state.wip || 0;
    const yld = (state.yieldRate || 1) * 100;
    const cost = state.costs ? state.costs.totalCost : 0;
    const maxPts = 500;

    // Extend live trend traces
    try {
        Plotly.extendTraces('live-chart-throughput', { y: [[tp]] }, [0], maxPts);
        Plotly.extendTraces('live-chart-wip', { y: [[wip]] }, [0], maxPts);
        Plotly.extendTraces('live-chart-yield', { y: [[yld]] }, [0], maxPts);
        Plotly.extendTraces('live-chart-cost', { y: [[cost]] }, [0], maxPts);
    } catch (e) {}

    // Accumulate SPC observations (sample every 20th frame = every ~1s at 60fps)
    if (liveUpdateCounter % 20 === 0) {
        liveThroughputObs.push(tp);
        liveYieldObs.push(yld);
        try {
            updateSPCChart('spc-chart-throughput', liveThroughputObs);
            updateSPCChart('spc-chart-yield', liveYieldObs);
        } catch (e) {}
    }

    // Update evidence panel
    if (liveUpdateCounter % 10 === 0) updateEvidencePanel(state);

    // Update SQDC Pareto (every 15th frame)
    if (liveUpdateCounter % 15 === 0) {
        try { updateSQDC(state); } catch (e) {}
    }
}

function updateSQDC(state) {
    if (!state || !state.stations) return;
    const machines = state.stations.filter(s => s.type === 'machine');
    if (machines.length === 0) return;

    const opts = { responsive: true, displayModeBar: false };
    const paretoLayout = {
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a', size: 8 },
        margin: { t: 28, b: 50, l: 30, r: 30 },
        xaxis: { gridcolor: 'rgba(255,255,255,0.05)', tickangle: -30 },
        yaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis2: { overlaying: 'y', side: 'right', range: [0, 105], showgrid: false, ticksuffix: '%' },
        showlegend: false,
        height: 230, autosize: true,
    };

    // Helper: build cumulative % line from sorted values
    function cumPct(vals) {
        const total = vals.reduce((s, v) => s + v, 0);
        if (total === 0) return vals.map(() => 0);
        let running = 0;
        return vals.map(v => { running += v; return (running / total) * 100; });
    }

    // Q — Quality Pareto: defects by station
    const qData = machines
        .map(s => ({ name: s.name, total: (s.stats.scrapped || 0) + (s.stats.reworked || 0) }))
        .filter(d => d.total > 0)
        .sort((a, b) => b.total - a.total);
    if (qData.length > 0) {
        const qVals = qData.map(d => d.total);
        Plotly.react('sqdc-quality', [
            { x: qData.map(d => d.name), y: qVals, type: 'bar', marker: { color: '#e74c3c' } },
            { x: qData.map(d => d.name), y: cumPct(qVals), type: 'scatter', mode: 'lines+markers', yaxis: 'y2', line: { color: '#f59e0b', width: 2 }, marker: { size: 4 } },
        ], { ...paretoLayout, title: { text: 'Q — Defects by Station', font: { size: 10, color: '#9aaa9a' } } }, opts);
    } else {
        Plotly.react('sqdc-quality', [], { ...paretoLayout, title: { text: 'Q — Defects by Station', font: { size: 10, color: '#9aaa9a' } }, annotations: [{ text: 'No defects', showarrow: false, font: { size: 11, color: '#666' } }] }, opts);
    }

    // D — Delivery Pareto: downtime minutes by station
    const dData = machines
        .map(s => ({ name: s.name, downMin: (s.stats.down || 0) / 60 }))
        .filter(d => d.downMin > 0.1)
        .sort((a, b) => b.downMin - a.downMin);
    if (dData.length > 0) {
        const dVals = dData.map(d => d.downMin);
        Plotly.react('sqdc-delivery', [
            { x: dData.map(d => d.name), y: dVals, type: 'bar', marker: { color: '#ef4444' } },
            { x: dData.map(d => d.name), y: cumPct(dVals), type: 'scatter', mode: 'lines+markers', yaxis: 'y2', line: { color: '#f59e0b', width: 2 }, marker: { size: 4 } },
        ], { ...paretoLayout, title: { text: 'D — Downtime (min) by Station', font: { size: 10, color: '#9aaa9a' } } }, opts);
    } else {
        Plotly.react('sqdc-delivery', [], { ...paretoLayout, title: { text: 'D — Downtime (min) by Station', font: { size: 10, color: '#9aaa9a' } }, annotations: [{ text: 'No downtime', showarrow: false, font: { size: 11, color: '#666' } }] }, opts);
    }

    // C — Cost Pareto: cost categories
    const costs = state.costs || {};
    const cData = [
        { name: 'Labor', val: costs.labor || 0 },
        { name: 'Overtime', val: costs.overtimeCost || 0 },
        { name: 'Scrap', val: costs.scrapWaste || 0 },
        { name: 'Holding', val: costs.holdingCost || 0 },
        { name: 'Material', val: costs.material || 0 },
    ].filter(c => c.val > 0).sort((a, b) => b.val - a.val);
    if (cData.length > 0) {
        const cVals = cData.map(d => d.val);
        Plotly.react('sqdc-cost', [
            { x: cData.map(d => d.name), y: cVals, type: 'bar', marker: { color: '#6366f1' } },
            { x: cData.map(d => d.name), y: cumPct(cVals), type: 'scatter', mode: 'lines+markers', yaxis: 'y2', line: { color: '#f59e0b', width: 2 }, marker: { size: 4 } },
        ], { ...paretoLayout, title: { text: 'C — Cost by Category', font: { size: 10, color: '#9aaa9a' } } }, opts);
    }
}

function updateSPCChart(chartId, obs) {
    SvendSPC.updateIChart(chartId, obs);
}

function updateEvidencePanel(state) {
    if (!state) return;

    // Sandbox mode: show live trajectory without targets
    if (!activeMission || !activeMission.scoring) {
        const tp = state.throughput || 0;
        const yld = state.yieldRate != null ? state.yieldRate : 1;
        const cost = state.costs ? state.costs.totalCost : 0;
        const cpu = state.costPerUnit || 0;
        let html = '<div style="font-size:0.75rem;font-weight:600;color:var(--text-primary);margin-bottom:6px;">Live Trajectory</div>';
        html += renderEvidenceBar('Throughput', Math.min(1, tp / 100), `${tp.toFixed(1)} units/hr`);
        html += renderEvidenceBar('Yield', yld, `${(yld * 100).toFixed(1)}%`);
        html += renderEvidenceBar('Bottleneck', state.bottleneckUtil || 0, `${state.bottleneckName || '—'} at ${((state.bottleneckUtil || 0) * 100).toFixed(0)}%`);
        const fmt = (v) => v >= 1000 ? `$${(v / 1000).toFixed(1)}k` : `$${v.toFixed(0)}`;
        html += renderEvidenceBar('Cost/Unit', cpu > 0 ? Math.min(1, 50 / cpu) : 0.5, cpu > 0 ? `${fmt(cpu)} per unit` : 'No output yet');
        document.getElementById('evidence-bars').innerHTML = html;
        return;
    }

    const scoring = activeMission.scoring;
    const weights = scoring.weights || {};
    const elapsed = state.elapsed || 0;
    const totalTime = des ? (des.runTime || 28800) : 28800;
    const progress = Math.min(1, elapsed / totalTime);

    let html = '';
    let combinedProb = 1;

    // For each objective, compute P(meeting target | data so far) using Bayesian approach:
    // Use current trajectory + time remaining to estimate posterior probability

    if (weights.throughput) {
        const target = weights.throughput.target;
        const current = state.throughput || 0;
        // Simple logistic estimate: how likely is current rate to produce target by end?
        // If we're on track (current >= target), high probability; below = drops fast
        const ratio = current / Math.max(target, 0.01);
        const p = bayesianObjectiveP(ratio, progress);
        combinedProb *= p;
        html += renderEvidenceBar('Throughput', p, `${current.toFixed(1)}/hr vs ${target}/hr target`);
    }

    if (weights.yield) {
        const target = weights.yield.target;
        const actual = state.yieldRate != null ? state.yieldRate : 1;
        const ratio = actual / Math.max(target, 0.01);
        const p = bayesianObjectiveP(ratio, progress);
        combinedProb *= p;
        html += renderEvidenceBar('Yield', p, `${(actual * 100).toFixed(1)}% vs ${(target * 100).toFixed(0)}% target`);
    }

    if (weights.response_time) {
        const target = weights.response_time.target;
        const alerts = des ? des.missionAlerts : [];
        const acked = alerts.filter(a => a.acknowledged);
        let avgResp = 0;
        if (acked.length > 0) {
            avgResp = acked.reduce((s, a) => s + (a.ackTime - a.time), 0) / acked.length;
        }
        const ratio = avgResp > 0 ? target / Math.max(avgResp, 1) : 1;
        const p = acked.length === 0 ? 0.5 : Math.min(1, Math.max(0, ratio > 1 ? 0.95 : ratio * 0.9));
        combinedProb *= p;
        html += renderEvidenceBar('Response Time', p, acked.length > 0 ? `avg ${avgResp.toFixed(0)}s vs ${target}s target` : 'No alerts acknowledged yet');
    }

    if (weights.cost) {
        const target = weights.cost.target;
        const actual = state.costs ? state.costs.totalCost : 0;
        // Project final cost based on burn rate
        const projectedCost = progress > 0.05 ? actual / progress : actual;
        const ratio = projectedCost > 0 ? target / Math.max(projectedCost, 1) : 1;
        const p = bayesianObjectiveP(ratio, progress);
        combinedProb *= p;
        const fmt = (v) => v >= 1000 ? `$${(v / 1000).toFixed(1)}k` : `$${v.toFixed(0)}`;
        html += renderEvidenceBar('Cost Control', p, `${fmt(actual)} spent, ~${fmt(projectedCost)} projected vs ${fmt(target)} target`);
    }

    // Combined mission success probability
    html += `<div class="evidence-combined">`;
    html += renderEvidenceBar('Mission Success', combinedProb, `Combined probability of meeting all objectives`, true);
    html += `</div>`;

    document.getElementById('evidence-bars').innerHTML = html;
}

function bayesianObjectiveP(ratio, progress) {
    // Bayesian-inspired posterior estimate:
    // Prior: uniform (0.5). Evidence: ratio of current/target performance.
    // As more time passes (progress → 1), evidence dominates prior.
    // Uses a sigmoid to map ratio to probability, sharpened by progress.
    const sharpness = 3 + progress * 12; // gets more decisive as sim progresses
    const logOdds = sharpness * (ratio - 1);
    return 1 / (1 + Math.exp(-logOdds));
}

function renderEvidenceBar(label, p, detail, isMain) {
    const pct = (p * 100).toFixed(0);
    const color = p >= 0.7 ? '#4a9f6e' : p >= 0.4 ? '#f59e0b' : '#e74c3c';
    const fontSize = isMain ? '0.8rem' : '0.75rem';
    const barHeight = isMain ? '18px' : '14px';
    const fontWeight = isMain ? '700' : '600';
    return `<div class="evidence-row" style="margin-bottom:${isMain ? '4' : '6'}px;">
        <div class="evidence-label" style="font-size:${fontSize};font-weight:${fontWeight};">${label}</div>
        <div class="evidence-bar-track" style="height:${barHeight};">
            <div class="evidence-bar-fill" style="width:${pct}%;background:${color};"></div>
            <div class="evidence-bar-text" style="line-height:${barHeight};">${pct}%</div>
        </div>
    </div>
    <div style="font-size:0.6rem;color:var(--text-dim);margin-bottom:4px;padding-left:128px;">${detail}</div>`;
}

function destroyLiveCharts() {
    liveChartsInitialized = false;
    liveChartsReady = false;
    liveThroughputObs = [];
    liveYieldObs = [];
    for (const id of ['live-chart-throughput', 'live-chart-wip', 'live-chart-yield', 'live-chart-cost', 'spc-chart-throughput', 'spc-chart-yield', 'sqdc-quality', 'sqdc-delivery', 'sqdc-cost']) {
        try { Plotly.purge(id); } catch (e) {}
    }
    document.getElementById('evidence-bars').innerHTML = '';
}

function showMissionTabs() {
    document.getElementById('tab-live-btn').style.display = '';
    document.getElementById('tab-spc-btn').style.display = '';
    document.getElementById('tab-evidence-btn').style.display = '';
    document.getElementById('tab-sqdc-btn').style.display = '';
}

function hideMissionTabs() {
    document.getElementById('tab-live-btn').style.display = 'none';
    document.getElementById('tab-spc-btn').style.display = 'none';
    document.getElementById('tab-evidence-btn').style.display = 'none';
    document.getElementById('tab-sqdc-btn').style.display = 'none';
}

// =============================================================================
// Mission Loading / Briefing / AAR
// =============================================================================

function loadMission(scenarioId) {
    const scenario = (typeof GUIDED_SCENARIOS !== 'undefined' ? GUIDED_SCENARIOS : []).find(s => s.id === scenarioId);
    if (!scenario || scenario.mode !== 'mission') return;

    // Populate briefing overlay
    document.getElementById('briefing-title').textContent = scenario.title;
    document.getElementById('briefing-subtitle').textContent = scenario.overview || '';
    document.getElementById('briefing-text').textContent = scenario.briefing || '';

    const objEl = document.getElementById('briefing-objectives');
    if (scenario.scoring && scenario.scoring.objectives) {
        objEl.innerHTML = '<div style="font-weight:600;margin-bottom:6px;color:var(--text-primary);">Objectives</div>' +
            scenario.scoring.objectives.map(o => `<div style="color:var(--text-secondary);font-size:0.8rem;padding:2px 0;">&#8226; ${o}</div>`).join('');
    } else {
        objEl.innerHTML = '';
    }

    activeMission = scenario;
    document.getElementById('mission-briefing').classList.add('active');
}

function closeBriefing() {
    document.getElementById('mission-briefing').classList.remove('active');
    activeMission = null;
}

function startMission() {
    if (!activeMission) return;
    const scenario = activeMission;

    // Close briefing
    document.getElementById('mission-briefing').classList.remove('active');

    // Stop any running sim
    if (des && des.running) {
        des.running = false;
        if (animFrameId) cancelAnimationFrame(animFrameId);
    }
    des = null;
    currentSimId = null;

    // Deep-clone and load layout
    const sl = JSON.parse(JSON.stringify(scenario.layout));
    layout.stations = sl.stations || [];
    layout.sources = sl.sources || [];
    layout.sinks = sl.sinks || [];
    layout.connections = sl.connections || [];
    layout.work_centers = sl.work_centers || [];
    layout.operators = sl.operators || [];
    layout.utility_systems = sl.utility_systems || [];
    layout.shared_tools = sl.shared_tools || [];

    document.getElementById('sim-name').value = `Mission: ${scenario.title}`;

    // Apply config
    if (scenario.config) {
        if (scenario.config.warmup != null) document.getElementById('cfg-warmup').value = scenario.config.warmup;
        if (scenario.config.runtime != null) document.getElementById('cfg-runtime').value = scenario.config.runtime;
        if (scenario.config.speed != null) {
            document.getElementById('cfg-speed').value = scenario.config.speed;
            document.getElementById('cfg-speed-label').textContent = scenario.config.speed + 'x';
        }
        if (scenario.config.maint_crew_size != null) document.getElementById('cfg-maint-crew').value = scenario.config.maint_crew_size;
        if (scenario.config.agv_fleet_size != null) document.getElementById('cfg-agv-fleet').value = scenario.config.agv_fleet_size;
    }

    // Reset nextId past scenario IDs
    const allEls = [...layout.stations, ...layout.sources, ...layout.sinks, ...layout.connections];
    for (const el of allEls) {
        const num = parseInt(String(el.id).split('-').pop());
        if (!isNaN(num) && num >= nextId) nextId = num + 1;
    }

    // Clear sandbox scenario state
    activeScenario = null;
    activeStepIndex = 0;
    scenarioChallengeResults = null;
    savedRuns = [];

    renderCanvas();
    if (typeof renderOperatorList === 'function') renderOperatorList();
    if (typeof renderUtilityList === 'function') renderUtilityList();
    if (typeof renderToolList === 'function') renderToolList();
    resetView();
    hideTeachingPanel();
    closeScenarioLauncher();

    // Create DES with mission timeline
    const timelineCopy = JSON.parse(JSON.stringify(scenario.timeline || []));
    des = new PlantDES(layout);
    des.missionMode = true;
    des.missionTimeline = timelineCopy;

    // Schedule timeline events into the already-built heap
    for (const te of des.missionTimeline) {
        te.fired = false;
        des.eventQueue.push({
            time: des.warmupTime + te.at,
            type: 'mission_event',
            timelineEntry: te,
        });
    }

    // Set up callbacks
    des.onAlert = (alert) => {
        renderAlertBar();
        showToast(`⚠ ${alert.message}`, alert.severity === 'critical' ? 4000 : 2500);
    };
    des.onPause = () => {
        document.getElementById('btn-play').innerHTML = '&#9654; Play';
        updateCommandPanel();
    };

    // Show command panel, hide teaching panel
    showCommandPanel();
    missionStartTime = Date.now();

    // Init live dashboard
    showMissionTabs();
    initLiveCharts();
    const metricsPanel = document.getElementById('metrics-panel');
    metricsPanel.classList.add('open');
    switchMetricsTab('live');
    document.getElementById('metrics-toggle').textContent = '▼ Charts';

    // Auto-start simulation
    const speed = parseFloat(document.getElementById('cfg-speed').value) || 20;
    document.getElementById('btn-play').innerHTML = '&#9646;&#9646; Pause';

    des.runAnimated(speed, (state, done) => {
        updateLiveMetrics(state);
        renderCanvas();
        if (activeMission) {
            renderAlertBar();
            updateCommandPanel();
            updateLiveCharts(state);
        }
        if (done) {
            document.getElementById('btn-play').innerHTML = '&#9654; Play';
            const results = des.getResults();
            showResults(results);
            if (des.missionMode) showAfterActionReview(results);
        }
    });

    showToast(`Mission started: ${scenario.title}`);
}

function exitMission() {
    if (!activeMission) return;
    if (!confirm('Abort mission? Your progress will be lost.')) return;

    if (des && des.running) {
        des.running = false;
        if (animFrameId) cancelAnimationFrame(animFrameId);
    }
    des = null;

    hideCommandPanel();
    hideMissionTabs();
    destroyLiveCharts();
    switchMetricsTab('charts');
    document.getElementById('alert-bar').style.display = 'none';
    document.getElementById('alert-bar').innerHTML = '';
    document.getElementById('btn-play').innerHTML = '&#9654; Play';
    activeMission = null;
    missionStartTime = null;
    showToast('Mission aborted.');
}

function showAfterActionReview(results) {
    const scenario = activeMission;
    if (!scenario || !scenario.scoring) return;

    // Show AAR tab
    document.getElementById('tab-aar-btn').style.display = '';
    switchMetricsTab('aar');

    const scoring = scenario.scoring;
    const weights = scoring.weights || {};

    // Calculate individual scores
    const scores = {};
    let totalPoints = 0;
    let earnedPoints = 0;

    if (weights.throughput) {
        const target = weights.throughput.target;
        const pts = weights.throughput.points;
        const actual = results.throughput || 0;
        const pct = Math.min(1, actual / target);
        scores.throughput = { label: 'Throughput', target, actual: actual.toFixed(1), pct, pts, earned: pct * pts };
        totalPoints += pts;
        earnedPoints += pct * pts;
    }
    if (weights.yield) {
        const target = weights.yield.target;
        const pts = weights.yield.points;
        const actual = results.overall_yield != null ? results.overall_yield : 1;
        const pct = Math.min(1, actual / target);
        scores.yield = { label: 'Yield', target: (target * 100).toFixed(0) + '%', actual: (actual * 100).toFixed(1) + '%', pct, pts, earned: pct * pts };
        totalPoints += pts;
        earnedPoints += pct * pts;
    }
    if (weights.response_time) {
        const target = weights.response_time.target;
        const pts = weights.response_time.points;
        // Avg time between alert fire and acknowledgment
        const alerts = results.mission_alerts || [];
        const acked = alerts.filter(a => a.acknowledged);
        let avgResponse = 0;
        if (acked.length > 0) {
            avgResponse = acked.reduce((s, a) => s + (a.ackTime - a.time), 0) / acked.length;
        }
        const pct = avgResponse <= target ? 1 : Math.max(0, 1 - (avgResponse - target) / target);
        scores.response_time = { label: 'Response Time', target: target + 's', actual: avgResponse.toFixed(0) + 's', pct, pts, earned: pct * pts };
        totalPoints += pts;
        earnedPoints += pct * pts;
    }
    if (weights.cost) {
        const target = weights.cost.target;
        const pts = weights.cost.points;
        const actual = results.total_cost || 0;
        const pct = actual <= target ? 1 : Math.max(0, 1 - (actual - target) / target);
        scores.cost = { label: 'Cost Control', target: '$' + target, actual: '$' + actual.toFixed(0), pct, pts, earned: pct * pts };
        totalPoints += pts;
        earnedPoints += pct * pts;
    }

    const compositeScore = totalPoints > 0 ? (earnedPoints / totalPoints) * 100 : 0;
    const grade = compositeScore >= 90 ? 'A' : compositeScore >= 80 ? 'B' : compositeScore >= 70 ? 'C' : compositeScore >= 60 ? 'D' : 'F';

    // Build unified timeline (events + decisions sorted by time)
    const timeline = [];
    for (const a of (results.mission_alerts || [])) {
        timeline.push({ time: a.time, type: 'event', severity: a.severity, text: a.message });
    }
    for (const d of (results.decision_log || [])) {
        timeline.push({ time: d.time, type: 'decision', text: `${d.action}: ${d.detail}` });
    }
    timeline.sort((a, b) => a.time - b.time);

    // Render AAR
    let html = '';

    // Grade header
    html += `<div style="text-align:center;padding:16px 0;">`;
    html += `<div style="font-size:2.5rem;font-weight:700;color:${grade === 'A' ? '#4a9f6e' : grade === 'B' ? '#3b82f6' : grade === 'C' ? '#f59e0b' : '#e74c3c'};">${grade}</div>`;
    html += `<div style="font-size:1.2rem;color:var(--text-secondary);">${compositeScore.toFixed(0)}%</div>`;
    html += `</div>`;

    // Score breakdown
    html += `<div style="margin-bottom:16px;">`;
    html += `<div style="font-weight:600;font-size:0.85rem;color:var(--text-primary);margin-bottom:8px;">Score Breakdown</div>`;
    for (const [key, s] of Object.entries(scores)) {
        html += `<div style="margin-bottom:8px;">`;
        html += `<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:var(--text-secondary);margin-bottom:2px;">`;
        html += `<span>${s.label}</span><span>${s.actual} / ${s.target} (${s.earned.toFixed(1)}/${s.pts})</span>`;
        html += `</div>`;
        html += `<div class="aar-score-bar"><div class="aar-score-fill" style="width:${(s.pct * 100).toFixed(0)}%;background:${s.pct >= 0.8 ? '#4a9f6e' : s.pct >= 0.6 ? '#f59e0b' : '#e74c3c'};"></div></div>`;
        html += `</div>`;
    }
    html += `</div>`;

    // Timeline
    if (timeline.length > 0) {
        html += `<div style="margin-bottom:16px;">`;
        html += `<div style="font-weight:600;font-size:0.85rem;color:var(--text-primary);margin-bottom:8px;">Mission Timeline</div>`;
        html += `<div class="aar-timeline">`;
        for (const entry of timeline) {
            const isEvent = entry.type === 'event';
            const dotColor = isEvent ? (entry.severity === 'critical' ? '#e74c3c' : entry.severity === 'warning' ? '#f59e0b' : '#3b82f6') : '#4a9f6e';
            html += `<div class="aar-timeline-entry">`;
            html += `<div class="aar-timeline-dot" style="background:${dotColor};"></div>`;
            html += `<div style="font-size:0.7rem;color:var(--text-dim);font-family:monospace;">${formatSimTime(entry.time)}</div>`;
            html += `<div style="font-size:0.75rem;color:var(--text-secondary);">${entry.text}</div>`;
            html += `</div>`;
        }
        html += `</div></div>`;
    }

    // Debrief notes
    if (scenario.debrief && scenario.debrief.length > 0) {
        html += `<div style="margin-bottom:16px;">`;
        html += `<div style="font-weight:600;font-size:0.85rem;color:var(--text-primary);margin-bottom:8px;">Debrief</div>`;
        for (const d of scenario.debrief) {
            html += `<div style="margin-bottom:10px;padding:8px;background:var(--bg-tertiary);border-radius:4px;">`;
            html += `<div style="font-weight:600;font-size:0.75rem;color:var(--text-primary);margin-bottom:4px;">${d.topic}</div>`;
            html += `<div style="font-size:0.75rem;color:var(--text-secondary);margin-bottom:4px;">${d.analysis}</div>`;
            html += `<div style="font-size:0.7rem;color:var(--accent);"><strong>Optimal:</strong> ${d.optimalAction}</div>`;
            html += `</div>`;
        }
        html += `</div>`;
    }

    document.getElementById('aar-content').innerHTML = html;

    // Save progress
    scenarioProgress[scenario.id] = {
        completed: true,
        score: Math.round(compositeScore),
        grade: grade,
        timestamp: Date.now(),
    };
    saveScenarioProgress();

    // Clean up mission state
    hideCommandPanel();
    destroyLiveCharts();
    document.getElementById('alert-bar').style.display = 'none';
    activeMission = null;
    missionStartTime = null;
}

// =============================================================================
// Save / Load / Import
// =============================================================================

async function saveLayout() {
    const name = document.getElementById('sim-name').value || 'Untitled Plant';
    const indicator = document.getElementById('save-indicator');
    indicator.textContent = '● Saving...';
    indicator.className = 'save-indicator saving';

    const body = {
        name,
        stations: layout.stations,
        connections: layout.connections,
        sources: layout.sources,
        sinks: layout.sinks,
        work_centers: layout.work_centers,
        utility_systems: layout.utility_systems || [],
        shared_tools: layout.shared_tools || [],
        simulation_config: {
            warmup_time: +document.getElementById('cfg-warmup').value,
            run_time: +document.getElementById('cfg-runtime').value,
            speed_factor: +document.getElementById('cfg-speed').value,
            calloff_rate: +document.getElementById('cfg-calloff').value,
            quit_rate: +document.getElementById('cfg-quit').value,
            labor_cost_per_hour: +document.getElementById('cfg-labor-cost').value,
            ot_premium: +document.getElementById('cfg-ot-premium').value,
            holding_cost_per_unit_hour: +document.getElementById('cfg-holding-cost').value,
            maint_crew_size: +document.getElementById('cfg-maint-crew').value,
            agv_fleet_size: +document.getElementById('cfg-agv-fleet').value,
            mgmt_reactivity: document.getElementById('cfg-mgmt-reactivity').value,
            ot_wip_threshold: +document.getElementById('cfg-ot-wip-threshold').value,
            ot_max_hours: +document.getElementById('cfg-ot-max-hours').value,
            revenue_per_unit: +document.getElementById('cfg-revenue-per-unit').value,
            eco_rate: +document.getElementById('cfg-eco-rate').value,
            inspector_pool: +document.getElementById('cfg-inspector-pool').value,
            operators: layout.operators,
        },
        zoom: canvasZoom,
        pan_x: canvasPanX,
        pan_y: canvasPanY,
    };

    try {
        let resp;
        if (currentSimId) {
            resp = await fetch(`/api/plantsim/${currentSimId}/update/`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCsrf() },
                body: JSON.stringify(body),
            });
        } else {
            resp = await fetch('/api/plantsim/create/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCsrf() },
                body: JSON.stringify(body),
            });
        }
        const data = await resp.json();
        if (data.simulation) {
            currentSimId = data.simulation.id || data.id;
            // Update URL without reload
            if (!window.location.pathname.includes(currentSimId)) {
                history.replaceState(null, '', `/app/simulator/${currentSimId}/`);
            }
        }
        indicator.textContent = '● Saved';
        indicator.className = 'save-indicator saved';
    } catch (e) {
        indicator.textContent = '● Save failed';
        indicator.className = 'save-indicator';
    }
}

async function loadLayoutList() {
    try {
        const resp = await fetch('/api/plantsim/', { headers: { 'X-CSRFToken': getCsrf() } });
        const data = await resp.json();
        const sims = data.simulations || [];
        if (sims.length === 0) { showToast('No saved simulations'); return; }

        const panel = document.getElementById('props-panel');
        const content = document.getElementById('props-content');
        panel.classList.add('active');
        window._pendingSims = sims;
        content.innerHTML = `
            <div style="font-size:0.8rem;font-weight:600;margin-bottom:6px;">Load Simulation</div>
            <div class="prop-row"><span class="prop-label">Layout</span>
                <select class="prop-input" id="load-sim-select">
                    ${sims.map((s, i) => `<option value="${i}">${s.name}</option>`).join('')}
                </select>
            </div>
            <button class="sim-btn" onclick="executeLoadSim()" style="width:100%;margin-top:6px;">Load</button>
        `;
        return;
    } catch (e) { showToast('Failed to load list'); }
}

function executeLoadSim() {
    const idx = +document.getElementById('load-sim-select')?.value;
    if (window._pendingSims && window._pendingSims[idx]) {
        loadSimulation(window._pendingSims[idx]);
        document.getElementById('props-panel').classList.remove('active');
    }
}

function loadSimulation(sim) {
    // Exit any active scenario when loading a saved simulation
    if (activeScenario) exitScenario();

    currentSimId = sim.id;
    layout = {
        stations: sim.stations || [],
        connections: sim.connections || [],
        sources: sim.sources || [],
        sinks: sim.sinks || [],
        work_centers: sim.work_centers || [],
        utility_systems: sim.utility_systems || [],
        shared_tools: sim.shared_tools || [],
    };
    document.getElementById('sim-name').value = sim.name || 'Untitled Plant';
    if (sim.simulation_config) {
        document.getElementById('cfg-warmup').value = sim.simulation_config.warmup_time || 300;
        document.getElementById('cfg-runtime').value = sim.simulation_config.run_time || 3600;
        document.getElementById('cfg-speed').value = sim.simulation_config.speed_factor || 20;
        if (sim.simulation_config.operators) {
            layout.operators = sim.simulation_config.operators;
        }
        if (sim.simulation_config.calloff_rate != null) {
            document.getElementById('cfg-calloff').value = sim.simulation_config.calloff_rate;
        }
        if (sim.simulation_config.quit_rate != null) {
            document.getElementById('cfg-quit').value = sim.simulation_config.quit_rate;
        }
        if (sim.simulation_config.labor_cost_per_hour != null) {
            document.getElementById('cfg-labor-cost').value = sim.simulation_config.labor_cost_per_hour;
        }
        if (sim.simulation_config.ot_premium != null) {
            document.getElementById('cfg-ot-premium').value = sim.simulation_config.ot_premium;
        }
        if (sim.simulation_config.holding_cost_per_unit_hour != null) {
            document.getElementById('cfg-holding-cost').value = sim.simulation_config.holding_cost_per_unit_hour;
        }
        if (sim.simulation_config.maint_crew_size != null) {
            document.getElementById('cfg-maint-crew').value = sim.simulation_config.maint_crew_size;
        }
        if (sim.simulation_config.agv_fleet_size != null) {
            document.getElementById('cfg-agv-fleet').value = sim.simulation_config.agv_fleet_size;
        }
        if (sim.simulation_config.mgmt_reactivity) {
            document.getElementById('cfg-mgmt-reactivity').value = sim.simulation_config.mgmt_reactivity;
        }
        if (sim.simulation_config.ot_wip_threshold != null) {
            document.getElementById('cfg-ot-wip-threshold').value = sim.simulation_config.ot_wip_threshold;
        }
        if (sim.simulation_config.ot_max_hours != null) {
            document.getElementById('cfg-ot-max-hours').value = sim.simulation_config.ot_max_hours;
        }
        if (sim.simulation_config.revenue_per_unit != null) {
            document.getElementById('cfg-revenue-per-unit').value = sim.simulation_config.revenue_per_unit;
        }
        if (sim.simulation_config.eco_rate != null) {
            document.getElementById('cfg-eco-rate').value = sim.simulation_config.eco_rate;
        }
        if (sim.simulation_config.inspector_pool != null) {
            document.getElementById('cfg-inspector-pool').value = sim.simulation_config.inspector_pool;
        }
    }
    canvasZoom = sim.zoom || 1;
    canvasPanX = sim.pan_x || 0;
    canvasPanY = sim.pan_y || 0;

    // Ensure nextId is beyond existing IDs
    const allIds = [...layout.sources, ...layout.stations, ...layout.sinks, ...layout.connections].map(e => e.id);
    for (const id of allIds) {
        const num = parseInt(id.split('-').pop());
        if (!isNaN(num) && num >= nextId) nextId = num + 1;
    }

    des = null;

    // Load saved runs from server for comparison
    savedRuns = [];
    if (sim.simulation_results && sim.simulation_results.length > 0) {
        sim.simulation_results.forEach((r, i) => {
            savedRuns.push({
                label: `${sim.name || 'Run'} #${i + 1}`,
                timestamp: 0,
                results: r,
            });
        });
        updateRunSelectors();
    }

    history.replaceState(null, '', `/app/simulator/${currentSimId}/`);
    renderCanvas();
    renderOperatorList();
    if (typeof renderUtilityList === 'function') renderUtilityList();
    if (typeof renderToolList === 'function') renderToolList();
    document.getElementById('save-indicator').textContent = '● Saved';
    document.getElementById('save-indicator').className = 'save-indicator saved';
}

async function importFromVSM() {
    if (!currentSimId) {
        showToast('Save the simulation first, then import');
        return;
    }
    try {
        const resp = await fetch('/api/vsm/', { headers: { 'X-CSRFToken': getCsrf() } });
        const data = await resp.json();
        const maps = data.maps || [];
        if (maps.length === 0) { showToast('No VSMs found'); return; }

        const panel = document.getElementById('props-panel');
        const content = document.getElementById('props-content');
        panel.classList.add('active');
        window._pendingVSMs = maps;
        window._pendingSimId = currentSimId;
        content.innerHTML = `
            <div style="font-size:0.8rem;font-weight:600;margin-bottom:6px;">Import from VSM</div>
            <div class="prop-row"><span class="prop-label">VSM</span>
                <select class="prop-input" id="import-vsm-select">
                    ${maps.map((m, i) => `<option value="${i}">${m.name}</option>`).join('')}
                </select>
            </div>
            <button class="sim-btn" onclick="executeVSMImport()" style="width:100%;margin-top:6px;">Import</button>
        `;
        return;
    } catch (e) { showToast('Import failed'); }
}

async function executeVSMImport() {
    const idx = +document.getElementById('import-vsm-select')?.value;
    const maps = window._pendingVSMs;
    const simId = window._pendingSimId;
    if (!maps || !maps[idx] || !simId) return;
    try {
        const importResp = await fetch(`/api/plantsim/${simId}/import-vsm/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCsrf() },
            body: JSON.stringify({ vsm_id: maps[idx].id }),
        });
        const result = await importResp.json();
        if (result.simulation) {
            loadSimulation(result.simulation);
            showToast(`Imported ${result.imported_stations} stations from VSM`);
        }
        document.getElementById('props-panel').classList.remove('active');
    } catch (e) { showToast('Import failed'); }
}
