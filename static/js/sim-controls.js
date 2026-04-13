// =============================================================================
// Simulation Controls
// =============================================================================

function toggleSimulation() {
    // Mission mode: handle pause/resume via mission system
    if (des && des.missionMode && des.paused) {
        resumeMission();
        return;
    }

    if (des && des.running) {
        des.running = false;
        if (animFrameId) cancelAnimationFrame(animFrameId);
        document.getElementById('btn-play').innerHTML = '&#9654; Play';
        return;
    }

    if (!des) {
        des = new PlantDES(layout);
    }

    // Show live dashboard tabs for any animated run
    showMissionTabs();
    initLiveCharts();
    const metricsPanel = document.getElementById('metrics-panel');
    metricsPanel.classList.add('open');
    switchMetricsTab('live');
    document.getElementById('metrics-toggle').textContent = '▼ Charts';

    const speed = parseFloat(document.getElementById('cfg-speed').value) || 20;
    document.getElementById('btn-play').innerHTML = '&#9646;&#9646; Pause';

    des.runAnimated(speed, (state, done) => {
        updateLiveMetrics(state);
        renderCanvas();
        updateLiveCharts(state);
        if (activeMission) {
            renderAlertBar();
            updateCommandPanel();
        }
        if (done) {
            document.getElementById('btn-play').innerHTML = '&#9654; Play';
            const results = des.getResults();
            showResults(results);
            if (des.missionMode) showAfterActionReview(results);
        }
    });
}

function stepSimulation() {
    if (!des) des = new PlantDES(layout);
    const state = des.step();
    updateLiveMetrics(state);
    renderCanvas();
}

function fastForward() {
    des = new PlantDES(layout);
    const results = des.runToCompletion();
    updateLiveMetrics(des.getState());
    renderCanvas();
    showResults(results);
}

function resetSimulation() {
    if (des && des.running) {
        des.running = false;
        if (animFrameId) cancelAnimationFrame(animFrameId);
    }
    des = null;
    document.getElementById('btn-play').innerHTML = '&#9654; Play';
    document.getElementById('m-throughput').textContent = '—';
    document.getElementById('m-wip').textContent = '—';
    document.getElementById('m-leadtime').textContent = '—';
    document.getElementById('m-bottleneck').textContent = '—';
    document.getElementById('m-yield').textContent = '—';
    document.getElementById('m-scrap').textContent = '—';
    document.getElementById('m-littles').textContent = 'L = λW check: —';
    document.getElementById('m-ontime').textContent = '—';
    document.getElementById('m-rush').textContent = '—';
    document.getElementById('m-rush').style.color = '';
    document.getElementById('m-escaped').textContent = '—';
    document.getElementById('m-escaped').style.color = '';
    document.getElementById('m-total-cost').textContent = '—';
    document.getElementById('m-cost-per-unit').textContent = '—';
    document.getElementById('m-cust-sat').textContent = '—';
    document.getElementById('m-cust-sat').style.color = '';
    document.getElementById('m-net-revenue').textContent = '—';
    document.getElementById('m-net-revenue').style.color = '';
    destroyLiveCharts();
    hideMissionTabs();
    switchMetricsTab('charts');
    renderCanvas();
}

function updateLiveMetrics(state) {
    document.getElementById('m-throughput').textContent = `${state.throughput.toFixed(1)}/hr`;
    document.getElementById('m-wip').textContent = state.wip.toFixed(1);
    document.getElementById('m-leadtime').textContent = `${state.avgLeadTime.toFixed(1)}s`;
    document.getElementById('m-bottleneck').textContent = state.bottleneckName;

    // Quality metrics
    document.getElementById('m-yield').textContent = `${(state.yieldRate * 100).toFixed(1)}%`;
    document.getElementById('m-yield').style.color = state.yieldRate >= 0.95 ? 'var(--accent-primary)' : state.yieldRate >= 0.85 ? '#f59e0b' : '#e74c3c';
    document.getElementById('m-scrap').textContent = `${state.scrapped} / ${state.reworked}`;
    document.getElementById('m-changeovers').textContent = state.changeovers || 0;

    // Service level & stockouts
    if (state.serviceLevel != null) {
        const sl = state.serviceLevel * 100;
        document.getElementById('m-service-level').textContent = `${sl.toFixed(1)}%`;
        document.getElementById('m-service-level').style.color = sl >= 95 ? 'var(--accent-primary)' : sl >= 85 ? '#f59e0b' : '#e74c3c';
    }
    document.getElementById('m-stockouts').textContent = state.totalStockouts || 0;
    if (state.totalStockouts > 0) {
        document.getElementById('m-stockouts').style.color = '#e74c3c';
    }

    // Rush order / on-time metrics
    if (state.onTimeDelivery != null) {
        const otd = state.onTimeDelivery * 100;
        document.getElementById('m-ontime').textContent = `${otd.toFixed(1)}%`;
        document.getElementById('m-ontime').style.color = otd >= 95 ? 'var(--accent-primary)' : otd >= 80 ? '#f59e0b' : '#e74c3c';
    }
    const rushLabel = `${state.rushOrderCount || 0} / ${state.lateOrderCount || 0}`;
    document.getElementById('m-rush').textContent = rushLabel;
    if (state.promotedToRush > 0) {
        document.getElementById('m-rush').style.color = '#e74c3c';
        document.getElementById('m-rush').title = `${state.promotedToRush} orders promoted to rush (expediting spiral)`;
    }

    // Escaped defect metrics
    const escapedLabel = `${state.escapedDefects || 0} / ${state.customerReturns || 0}`;
    document.getElementById('m-escaped').textContent = escapedLabel;
    if (state.customerReturns > 0) {
        document.getElementById('m-escaped').style.color = '#e74c3c';
        document.getElementById('m-escaped').title = `${state.detectedDownstream || 0} caught downstream, ${state.customerReturns} reached customer`;
    }

    // Cost accounting
    if (state.costs) {
        const fmt = (v) => v >= 1000 ? `$${(v/1000).toFixed(1)}k` : `$${v.toFixed(0)}`;
        document.getElementById('m-total-cost').textContent = fmt(state.costs.totalCost);
        const breakdown = `Labor: ${fmt(state.costs.labor)} | OT: ${fmt(state.costs.overtimeCost)} | Material: ${fmt(state.costs.material)} | Scrap: ${fmt(state.costs.scrapWaste)} | Holding: ${fmt(state.costs.holdingCost)}`;
        document.getElementById('m-total-cost').title = breakdown;
        document.getElementById('m-cost-per-unit').textContent = state.costPerUnit > 0
            ? `$${state.costPerUnit.toFixed(2)}` : '—';
    }

    // Overtime status
    const otEl = document.getElementById('m-overtime');
    if (state.overtimeActive) {
        otEl.textContent = 'ACTIVE';
        otEl.style.color = '#f59e0b';
        otEl.title = `${state.overtimeShifts} OT shifts | ${state.totalOTHours.toFixed(0)} total OT hours`;
    } else if (state.overtimeShifts > 0) {
        otEl.textContent = `${state.overtimeShifts} shifts`;
        otEl.style.color = 'var(--text-dim)';
        otEl.title = `${state.totalOTHours.toFixed(0)} total OT hours (not currently active)`;
    } else {
        otEl.textContent = 'Off';
        otEl.style.color = 'var(--text-dim)';
    }

    // AGV fleet metrics
    if (state.agvFleet && state.agvFleet.size > 0) {
        const agv = state.agvFleet;
        document.getElementById('m-agv-queue').textContent = `${agv.queueLength} waiting`;
        document.getElementById('m-agv-queue').style.color = agv.queueLength > 0 ? '#f59e0b' : 'var(--accent-primary)';
        document.getElementById('m-agv-queue').title = `${agv.available}/${agv.size} AGVs free | Avg wait: ${agv.avgWaitTime.toFixed(0)}s | ${agv.tripsCompleted} trips`;
    } else {
        document.getElementById('m-agv-queue').textContent = '∞';
        document.getElementById('m-agv-queue').title = 'Unlimited transport (set AGV fleet > 0)';
    }

    // Expired WIP
    const expEl = document.getElementById('m-expired-wip');
    expEl.textContent = state.expiredWIP || 0;
    if (state.expiredWIP > 0) expEl.style.color = '#e74c3c';

    // Utility system live status — refresh the sidebar display
    if (typeof renderUtilityList === 'function' && state.utilitySystems && state.utilitySystems.length > 0) {
        renderUtilityList();
    }

    // Maintenance crew metrics
    if (state.maintenanceCrew && state.maintenanceCrew.size > 0) {
        const mc = state.maintenanceCrew;
        document.getElementById('m-maint-queue').textContent = `${mc.queueLength} waiting`;
        document.getElementById('m-maint-queue').style.color = mc.queueLength > 0 ? '#e74c3c' : 'var(--accent-primary)';
        document.getElementById('m-maint-queue').title = `${mc.available}/${mc.size} techs free | Avg wait: ${mc.avgWaitTime.toFixed(0)}s | ${mc.repairsCompleted} repairs done`;
    } else {
        document.getElementById('m-maint-queue').textContent = '∞';
        document.getElementById('m-maint-queue').title = 'Unlimited crew (set crew size > 0 to enable contention)';
    }

    // Customer behavior metrics
    const satEl = document.getElementById('m-cust-sat');
    const revEl = document.getElementById('m-net-revenue');
    if (state.customerSatisfaction != null && state.totalRevenue > 0) {
        const satPct = (state.customerSatisfaction * 100).toFixed(0);
        satEl.textContent = `${satPct}%`;
        satEl.style.color = state.customerSatisfaction < 0.5 ? '#e74c3c' : state.customerSatisfaction < 0.8 ? '#f39c12' : 'var(--accent-primary)';
        satEl.title = `${state.customerLostOrders || 0} lost orders | ${state.totalReturnCost.toFixed(0)} return costs`;
        const net = state.netRevenue || 0;
        revEl.textContent = `$${net >= 0 ? '' : '-'}${Math.abs(net).toFixed(0)}`;
        revEl.style.color = net < 0 ? '#e74c3c' : 'var(--accent-primary)';
        revEl.title = `Revenue: $${state.totalRevenue.toFixed(0)} | Returns: -$${state.totalReturnCost.toFixed(0)} | Costs: -$${(state.costs?.totalCost || 0).toFixed(0)}`;
    } else {
        satEl.textContent = '—';
        satEl.style.color = '';
        revEl.textContent = '—';
        revEl.style.color = '';
    }

    // Update operator list display during sim
    if (des && des.operators.length > 0) {
        // Sync DES operator state back to layout for rendering
        for (const desOp of des.operators) {
            const layoutOp = layout.operators.find(o => o.id === desOp.id);
            if (layoutOp) {
                layoutOp.status = desOp.status;
                layoutOp.assignedTo = desOp.assignedTo;
                layoutOp.skills = { ...desOp.skills };
            }
        }
        renderOperatorList();
    }

    // Little's Law check: L = λ × W
    const lambda = state.throughput / 3600;
    const predicted = lambda * state.avgLeadTime;
    const actual = state.wip;
    const match = actual > 0 ? Math.abs(predicted - actual) / actual < 0.3 : true;
    document.getElementById('m-littles').textContent = `L=λW: ${predicted.toFixed(1)} vs ${actual.toFixed(1)} ${match ? '✓' : '✗'}`;
    document.getElementById('m-littles').style.color = match ? 'var(--accent-primary)' : '#e74c3c';
}

// =============================================================================
// Results Charts
// =============================================================================

function showResults(results) {
    const panel = document.getElementById('metrics-panel');
    panel.classList.add('open');

    // Switch to charts tab so post-run plots are visible
    switchMetricsTab('charts');
    document.getElementById('metrics-toggle').textContent = '▼ Charts';

    // Auto-save run for comparison
    saveRun(results);

    // Queue lengths over time
    const queueTraces = [];
    for (const [id, hist] of Object.entries(results.station_history || {})) {
        queueTraces.push({
            y: hist.queue,
            name: hist.name,
            type: 'scatter',
            mode: 'lines',
        });
    }
    Plotly.newPlot('chart-queues', queueTraces, {
        title: { text: 'Queue Lengths', font: { size: 12, color: '#9aaa9a' } },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a', size: 10 },
        margin: { t: 30, b: 30, l: 35, r: 10 },
        xaxis: { title: 'Sample', gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis: { title: 'Queue', gridcolor: 'rgba(255,255,255,0.05)' },
        legend: { orientation: 'h', y: -0.2 },
    }, { responsive: true, displayModeBar: false });

    // Utilization stacked bars
    const stnNames = [];
    const processing = [], setup = [], down = [], starved = [], blocked = [], idle = [], onBreak = [];
    for (const [id, util] of Object.entries(results.station_utilizations || {})) {
        stnNames.push(util.name);
        processing.push((util.processing * 100).toFixed(1));
        setup.push((util.setup * 100).toFixed(1));
        down.push((util.down * 100).toFixed(1));
        starved.push((util.starved * 100).toFixed(1));
        blocked.push((util.blocked * 100).toFixed(1));
        idle.push((util.idle * 100).toFixed(1));
        onBreak.push(((util.onBreak || 0) * 100).toFixed(1));
    }

    Plotly.newPlot('chart-utilization', [
        { x: stnNames, y: processing, name: 'Processing', type: 'bar', marker: { color: '#4a9f6e' } },
        { x: stnNames, y: blocked, name: 'Blocked', type: 'bar', marker: { color: '#e74c3c' } },
        { x: stnNames, y: starved, name: 'Starved', type: 'bar', marker: { color: '#f59e0b' } },
        { x: stnNames, y: down, name: 'Down', type: 'bar', marker: { color: '#ef4444' } },
        { x: stnNames, y: onBreak, name: 'Break', type: 'bar', marker: { color: '#6366f1' } },
        { x: stnNames, y: idle, name: 'Idle', type: 'bar', marker: { color: '#374151' } },
    ], {
        title: { text: 'Station Utilization %', font: { size: 12, color: '#9aaa9a' } },
        barmode: 'stack',
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a', size: 10 },
        margin: { t: 30, b: 40, l: 35, r: 10 },
        yaxis: { range: [0, 100], gridcolor: 'rgba(255,255,255,0.05)' },
        legend: { orientation: 'h', y: -0.3, font: { size: 9 } },
    }, { responsive: true, displayModeBar: false });

    // Lead time histogram
    if (results.lead_times && results.lead_times.length > 0) {
        Plotly.newPlot('chart-leadtime', [{
            x: results.lead_times,
            type: 'histogram',
            nbinsx: 30,
            marker: { color: 'rgba(74,159,110,0.6)' },
        }], {
            title: { text: 'Lead Time Distribution', font: { size: 12, color: '#9aaa9a' } },
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#9aaa9a', size: 10 },
            margin: { t: 30, b: 30, l: 35, r: 10 },
            xaxis: { title: 'Lead Time (s)', gridcolor: 'rgba(255,255,255,0.05)' },
            yaxis: { title: 'Count', gridcolor: 'rgba(255,255,255,0.05)' },
        }, { responsive: true, displayModeBar: false });
    }

    // Evaluate scenario challenges if in guided mode
    if (activeScenario) evaluateScenarioChallenges(results);

    // Show after-action review for mission mode
    if (des && des.missionMode && activeMission) showAfterActionReview(results);
}

function toggleMetrics() {
    const panel = document.getElementById('metrics-panel');
    panel.classList.toggle('open');
    document.getElementById('metrics-toggle').textContent = panel.classList.contains('open') ? '▼ Charts' : '▲ Charts';
}

// =============================================================================
// Scenario Comparison
// =============================================================================

function switchMetricsTab(tab) {
    document.querySelectorAll('.sv-tab').forEach(t => t.classList.remove('active'));
    const tabBtn = document.querySelector(`.sv-tab[onclick*="'${tab}'"]`);
    if (tabBtn) tabBtn.classList.add('active');
    document.getElementById('tab-charts').classList.toggle('hidden', tab !== 'charts');
    const compareEl = document.getElementById('tab-compare');
    compareEl.classList.toggle('active', tab === 'compare');
    for (const t of ['aar', 'live', 'spc', 'evidence', 'sqdc']) {
        const el = document.getElementById('tab-' + t);
        if (el) el.style.display = tab === t ? 'flex' : 'none';
    }
    if (tab === 'compare') renderComparison();
    // Resize Plotly charts after tab becomes visible
    if (tab === 'live' || tab === 'spc' || tab === 'sqdc') {
        requestAnimationFrame(() => {
            const container = document.getElementById('tab-' + tab);
            if (container) container.querySelectorAll('.live-chart').forEach(el => {
                if (el.data) Plotly.Plots.resize(el);
            });
        });
    }
}

function saveRun(results) {
    const name = document.getElementById('sim-name').value || 'Untitled';
    const runLabel = `${name} #${savedRuns.length + 1}`;
    savedRuns.push({
        label: runLabel,
        timestamp: Date.now(),
        results: results,
    });

    // Also save to server if we have a sim ID
    if (currentSimId) {
        fetch(`/api/plantsim/${currentSimId}/results/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCsrf() },
            body: JSON.stringify({ results }),
        }).catch(() => {});
    }

    updateRunSelectors();
}

function updateRunSelectors() {
    const count = savedRuns.length;
    document.getElementById('run-count').textContent = `${count} saved run${count !== 1 ? 's' : ''}`;

    for (const selId of ['compare-a', 'compare-b']) {
        const sel = document.getElementById(selId);
        const curVal = sel.value;
        sel.innerHTML = '<option value="">— select —</option>';
        savedRuns.forEach((run, i) => {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = run.label;
            sel.appendChild(opt);
        });
        sel.value = curVal;
    }

    // Auto-select latest two runs if both selectors empty and we have 2+
    if (count >= 2) {
        const a = document.getElementById('compare-a');
        const b = document.getElementById('compare-b');
        if (!a.value && !b.value) {
            a.value = count - 2;
            b.value = count - 1;
        }
    }
}

function renderComparison() {
    const idxA = document.getElementById('compare-a').value;
    const idxB = document.getElementById('compare-b').value;
    const tableEl = document.getElementById('compare-table');
    const chartEl = document.getElementById('compare-chart-overlay');

    if (idxA === '' || idxB === '') {
        tableEl.innerHTML = '<div style="padding:20px;color:var(--text-dim);font-size:0.8rem;">Select two runs to compare.</div>';
        chartEl.innerHTML = '';
        return;
    }

    const runA = savedRuns[idxA].results;
    const runB = savedRuns[idxB].results;
    const labelA = savedRuns[idxA].label;
    const labelB = savedRuns[idxB].label;

    // Build comparison table
    const metrics = [
        { name: 'Throughput', key: 'throughput', unit: '/hr', higherBetter: true },
        { name: 'Avg WIP', key: 'avg_wip', unit: '', higherBetter: false },
        { name: 'Avg Lead Time', key: 'avg_lead_time', unit: 's', higherBetter: false },
        { name: 'Completed', key: 'completed_count', unit: '', higherBetter: true },
        { name: 'Yield', key: 'yield_rate', unit: '', higherBetter: true, isPct: true },
        { name: 'Scrapped', key: 'total_scrapped', unit: '', higherBetter: false },
        { name: 'Reworked', key: 'total_reworked', unit: '', higherBetter: false },
        { name: 'Changeovers', key: 'total_changeovers', unit: '', higherBetter: false },
        { name: 'Service Level', key: 'service_level', unit: '', higherBetter: true, isPct: true },
        { name: 'Stockouts', key: 'total_stockouts', unit: '', higherBetter: false },
        { name: 'On-Time Delivery', key: 'on_time_delivery', unit: '', higherBetter: true, isPct: true },
        { name: 'Rush Orders', key: 'rush_order_count', unit: '', higherBetter: false },
        { name: 'Late Orders', key: 'late_order_count', unit: '', higherBetter: false },
        { name: 'Promoted to Rush', key: 'promoted_to_rush', unit: '', higherBetter: false },
        { name: 'Escaped Defects', key: 'escaped_defects', unit: '', higherBetter: false },
        { name: 'Customer Returns', key: 'customer_returns', unit: '', higherBetter: false },
        { name: 'Cust. Satisfaction', key: 'customer_satisfaction', unit: '%', higherBetter: true, pct: true },
        { name: 'Net Revenue', key: 'net_revenue', unit: '$', higherBetter: true },
        { name: 'Lost Orders', key: 'customer_lost_orders', unit: '', higherBetter: false },
        { name: 'Cost / Unit', key: 'cost_per_unit', unit: '$', higherBetter: false },
        { name: 'Bottleneck', key: 'bottleneck_station_name', unit: '', isText: true },
    ];

    let html = '<table>';
    html += `<tr><th>Metric</th><th>${labelA}</th><th>${labelB}</th><th>Delta</th></tr>`;
    for (const m of metrics) {
        const a = runA[m.key];
        const b = runB[m.key];
        if (m.isText) {
            html += `<tr><td>${m.name}</td><td>${a || '—'}</td><td>${b || '—'}</td><td class="delta-neutral">${a === b ? 'same' : 'changed'}</td></tr>`;
        } else if (m.isPct) {
            const aP = (a * 100), bP = (b * 100);
            const delta = bP - aP;
            const sign = delta > 0 ? '+' : '';
            const better = m.higherBetter ? delta > 0 : delta < 0;
            const cls = Math.abs(delta) < 0.1 ? 'delta-neutral' : (better ? 'delta-pos' : 'delta-neg');
            html += `<tr><td>${m.name}</td><td>${aP.toFixed(1)}%</td><td>${bP.toFixed(1)}%</td>`;
            html += `<td class="${cls}">${sign}${delta.toFixed(1)}pp</td></tr>`;
        } else {
            const delta = b - a;
            const pct = a !== 0 ? ((delta / Math.abs(a)) * 100) : 0;
            const sign = delta > 0 ? '+' : '';
            const better = m.higherBetter ? delta > 0 : delta < 0;
            const cls = Math.abs(pct) < 1 ? 'delta-neutral' : (better ? 'delta-pos' : 'delta-neg');
            html += `<tr><td>${m.name}</td><td>${(+a).toFixed(1)}${m.unit}</td><td>${(+b).toFixed(1)}${m.unit}</td>`;
            html += `<td class="${cls}">${sign}${delta.toFixed(1)} (${sign}${pct.toFixed(1)}%)</td></tr>`;
        }
    }

    // Per-station utilization comparison
    const stnsA = runA.station_utilizations || {};
    const stnsB = runB.station_utilizations || {};
    const allStnIds = new Set([...Object.keys(stnsA), ...Object.keys(stnsB)]);
    if (allStnIds.size > 0) {
        html += '<tr><td colspan="4" style="padding-top:10px;font-weight:600;color:var(--text-dim);font-size:0.65rem;text-transform:uppercase;">Station Utilization (processing %)</td></tr>';
        for (const id of allStnIds) {
            const a = stnsA[id]?.processing ?? 0;
            const b = stnsB[id]?.processing ?? 0;
            const name = stnsA[id]?.name || stnsB[id]?.name || id;
            const delta = b - a;
            const pct = (delta * 100);
            const sign = pct > 0 ? '+' : '';
            const cls = Math.abs(pct) < 1 ? 'delta-neutral' : (pct > 0 ? 'delta-pos' : 'delta-neg');
            html += `<tr><td>${name}</td><td>${(a * 100).toFixed(1)}%</td><td>${(b * 100).toFixed(1)}%</td>`;
            html += `<td class="${cls}">${sign}${pct.toFixed(1)}pp</td></tr>`;
        }
    }
    html += '</table>';
    tableEl.innerHTML = html;

    // Overlaid lead time histograms
    const traces = [];
    if (runA.lead_times?.length) {
        traces.push({
            x: runA.lead_times, type: 'histogram', name: labelA,
            opacity: 0.6, marker: { color: 'rgba(74,159,110,0.6)' }, nbinsx: 25,
        });
    }
    if (runB.lead_times?.length) {
        traces.push({
            x: runB.lead_times, type: 'histogram', name: labelB,
            opacity: 0.6, marker: { color: 'rgba(59,130,246,0.6)' }, nbinsx: 25,
        });
    }
    if (traces.length) {
        Plotly.newPlot(chartEl, traces, {
            title: { text: 'Lead Time Overlay', font: { size: 12, color: '#9aaa9a' } },
            barmode: 'overlay',
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#9aaa9a', size: 10 },
            margin: { t: 30, b: 30, l: 35, r: 10 },
            xaxis: { title: 'Lead Time (s)', gridcolor: 'rgba(255,255,255,0.05)' },
            yaxis: { title: 'Count', gridcolor: 'rgba(255,255,255,0.05)' },
            legend: { orientation: 'h', y: -0.25 },
        }, { responsive: true, displayModeBar: false });
    } else {
        chartEl.innerHTML = '<div style="padding:20px;color:var(--text-dim);font-size:0.75rem;">No lead time data to compare.</div>';
    }
}

