/**
 * calc-sim-quality.js — Quality & Inventory Simulators for Operations Workbench
 *
 * Load order: after svend-math.js, svend-charts.js, svend-sim-core.js
 * Extracted from: calculators.html (inline script)
 *
 * Provides:
 *   Safety Stock (s,Q) simulator with stochastic demand/lead time
 *   FMEA Monte Carlo RPN simulation
 *   SMED changeover reduction simulator
 *   Heijunka (production leveling) simulator
 */

// ============================================================================
// Safety Stock Simulator
// ============================================================================

const ssSimState = {
    running: false, interval: null, day: 0, inventory: 0,
    inTransit: [],  // [{qty, daysLeft}]
    history: { day: [], inventory: [], demand: [], stockout: [], orderPlaced: [], arrival: [] },
    totalDemand: 0, totalServed: 0, totalHoldCost: 0, totalStockoutCost: 0,
    stockoutDays: 0, ordersPlaced: 0,
    stockoutLog: [],  // [{day, demand, inventoryBefore, shortfall}]
    params: null,
};

function ssRandn() {
    return SvendMath.randn();
}

function ssComputeDerived() {
    const demand = parseFloat(document.getElementById('ss-demand').value) || 100;
    const demandStd = parseFloat(document.getElementById('ss-demand-std').value) || 0;
    const lead = parseFloat(document.getElementById('ss-lead').value) || 5;
    const leadStd = parseFloat(document.getElementById('ss-lead-std').value) || 0;
    const z = parseFloat(document.getElementById('ss-service').value) || 1.65;
    const combined = Math.sqrt(lead * demandStd * demandStd + demand * demand * leadStd * leadStd);
    const ss = z * combined;
    const rop = demand * lead + ss;
    return { ss, rop, z, demand, demandStd, lead, leadStd };
}

function ssUpdateDerived() {
    const d = ssComputeDerived();
    document.getElementById('ss-derived-ss').textContent = Math.round(d.ss) + ' units';
    document.getElementById('ss-derived-rop').textContent = Math.round(d.rop) + ' units';
    document.getElementById('ss-derived-z').textContent = d.z.toFixed(2);
    // Update what-if display
    ssUpdateMultiplier(document.getElementById('ss-multiplier').value);
}

function ssGetParams() {
    const d = ssComputeDerived();
    const orderQty = parseFloat(document.getElementById('ss-order-qty').value) || 1000;
    const initInvRaw = document.getElementById('ss-init-inv').value;
    const initInv = initInvRaw ? parseFloat(initInvRaw) : Math.round(d.rop + orderQty);
    const duration = Math.min(365, Math.max(30, parseInt(document.getElementById('ss-duration').value) || 180));
    const holdCost = parseFloat(document.getElementById('ss-hold-cost').value) || 0;
    const stockoutCost = parseFloat(document.getElementById('ss-stockout-cost').value) || 0;
    const multiplier = parseFloat(document.getElementById('ss-multiplier').value) / 100;
    const effectiveSS = d.ss * multiplier;
    const effectiveROP = d.demand * d.lead + effectiveSS;
    return {
        demand: d.demand, demandStd: d.demandStd, lead: d.lead, leadStd: d.leadStd,
        z: d.z, baseSS: d.ss, baseROP: d.rop, ss: effectiveSS, rop: effectiveROP,
        orderQty, initInv, duration, holdCost, stockoutCost
    };
}

function ssInit() {
    const p = ssGetParams();
    ssSimState.params = p;
    ssSimState.day = 0;
    ssSimState.inventory = p.initInv;
    ssSimState.inTransit = [];
    ssSimState.history = { day: [], inventory: [], demand: [], stockout: [], orderPlaced: [], arrival: [] };
    ssSimState.totalDemand = 0;
    ssSimState.totalServed = 0;
    ssSimState.totalHoldCost = 0;
    ssSimState.totalStockoutCost = 0;
    ssSimState.stockoutDays = 0;
    ssSimState.ordersPlaced = 0;
    ssSimState.stockoutLog = [];
    document.getElementById('ss-day-total').textContent = ' / ' + p.duration;
}

function ssStart() {
    if (ssSimState.running) {
        clearInterval(ssSimState.interval);
        ssSimState.running = false;
        document.getElementById('ss-start-btn').innerHTML = '&#9654; Resume';
        return;
    }
    if (ssSimState.day === 0) ssInit();
    ssSimState.running = true;
    document.getElementById('ss-start-btn').innerHTML = '&#10074;&#10074; Pause';
    ssSimState.interval = setInterval(ssTick, 100);
}

function ssReset() {
    if (ssSimState.interval) clearInterval(ssSimState.interval);
    ssSimState.running = false;
    ssSimState.day = 0;
    ssSimState.inventory = 0;
    ssSimState.inTransit = [];
    ssSimState.params = null;
    document.getElementById('ss-start-btn').innerHTML = '&#9654; Start';
    document.getElementById('ss-day-counter').textContent = '0';
    document.getElementById('ss-m-inventory').innerHTML = '—<span class="calc-result-unit"> units</span>';
    document.getElementById('ss-m-service').innerHTML = '—<span class="calc-result-unit">%</span>';
    document.getElementById('ss-m-stockouts').innerHTML = '0<span class="calc-result-unit"> days</span>';
    document.getElementById('ss-m-avg-inv').innerHTML = '—<span class="calc-result-unit"> units</span>';
    document.getElementById('ss-m-orders').textContent = '0';
    document.getElementById('ss-m-fillrate').innerHTML = '—<span class="calc-result-unit">%</span>';
    document.getElementById('ss-cost-holding').textContent = '$0';
    document.getElementById('ss-cost-stockout').textContent = '$0';
    document.getElementById('ss-cost-total').textContent = '$0';
    document.getElementById('ss-stockout-log').innerHTML = '<em>No stockouts yet.</em>';
    document.getElementById('ss-insights').innerHTML = '<em>Start simulation to generate insights...</em>';
    try { Plotly.purge('ss-chart'); } catch(e) {}
    try { Plotly.purge('ss-cost-chart'); } catch(e) {}
    ssUpdateDerived();
}

function ssTick() {
    const speed = parseInt(document.getElementById('ss-speed').value) || 5;
    for (let i = 0; i < speed; i++) {
        ssStep();
        if (ssSimState.day >= ssSimState.params.duration) {
            ssStop();
            return;
        }
    }
    ssUpdateChart();
    ssUpdateMetrics();
    ssUpdateCosts();
    if (ssSimState.day % 10 === 0) ssUpdateInsights();
}

function ssStep() {
    const p = ssSimState.params;
    ssSimState.day++;
    const day = ssSimState.day;

    // 1. Generate stochastic demand
    const demand = Math.max(0, p.demand + ssRandn() * p.demandStd);

    // 2. Receive in-transit orders that have arrived
    let received = 0;
    const stillInTransit = [];
    for (const o of ssSimState.inTransit) {
        o.daysLeft--;
        if (o.daysLeft <= 0) { received += o.qty; }
        else stillInTransit.push(o);
    }
    ssSimState.inTransit = stillInTransit;
    ssSimState.inventory += received;

    // 3. Consume inventory
    const inventoryBefore = ssSimState.inventory;
    ssSimState.inventory -= demand;
    let shortfall = 0;
    let stockoutFlag = false;
    if (ssSimState.inventory < 0) {
        shortfall = -ssSimState.inventory;
        ssSimState.inventory = 0;
        stockoutFlag = true;
        ssSimState.stockoutDays++;
        ssSimState.stockoutLog.push({
            day, demand: Math.round(demand),
            inventoryBefore: Math.round(inventoryBefore),
            shortfall: Math.round(shortfall)
        });
    }

    // 4. Accumulate
    ssSimState.totalDemand += demand;
    ssSimState.totalServed += (demand - shortfall);
    ssSimState.totalHoldCost += ssSimState.inventory * p.holdCost;
    ssSimState.totalStockoutCost += shortfall * p.stockoutCost;

    // 5. Check reorder: (s,Q) policy — one open order at a time
    const inTransitQty = ssSimState.inTransit.reduce((s, o) => s + o.qty, 0);
    const invPosition = ssSimState.inventory + inTransitQty;
    let orderPlacedFlag = false;
    if (invPosition <= p.rop && ssSimState.inTransit.length === 0) {
        const lt = Math.max(1, Math.round(p.lead + ssRandn() * p.leadStd));
        ssSimState.inTransit.push({ qty: p.orderQty, daysLeft: lt });
        ssSimState.ordersPlaced++;
        orderPlacedFlag = true;
    }

    // 6. Record history
    const h = ssSimState.history;
    h.day.push(day);
    h.inventory.push(Math.round(ssSimState.inventory));
    h.demand.push(Math.round(demand));
    h.stockout.push(stockoutFlag);
    h.orderPlaced.push(orderPlacedFlag);
    h.arrival.push(received > 0);
}

function ssUpdateChart() {
    const h = ssSimState.history;
    const p = ssSimState.params;
    if (h.day.length < 2) return;

    const invTrace = {
        x: h.day, y: h.inventory, type: 'scatter', mode: 'lines',
        name: 'Inventory', fill: 'tozeroy', fillcolor: 'rgba(39,174,96,0.12)',
        line: { color: '#27ae60', width: 2 }
    };

    const soDays = h.day.filter((_, i) => h.stockout[i]);
    const stockoutTrace = {
        x: soDays, y: soDays.map(() => 0),
        type: 'scatter', mode: 'markers', name: 'Stockout',
        marker: { color: '#e74c3c', size: 9, symbol: 'x' }
    };

    const arrDays = [], arrY = [];
    h.day.forEach((d, i) => { if (h.arrival[i]) { arrDays.push(d); arrY.push(h.inventory[i]); } });
    const arrivalTrace = {
        x: arrDays, y: arrY, type: 'scatter', mode: 'markers', name: 'Arrival',
        marker: { color: '#9b59b6', size: 8, symbol: 'triangle-up' }
    };

    const demandTrace = {
        x: h.day, y: h.demand, type: 'scatter', mode: 'lines',
        name: 'Demand', line: { color: '#3498db', width: 1, dash: 'dot' },
        yaxis: 'y2', opacity: 0.6
    };

    const maxDay = p.duration;
    const shapes = [
        { type: 'line', x0: 1, x1: maxDay, y0: p.rop, y1: p.rop,
          line: { color: '#e74c3c', width: 1.5, dash: 'dash' } },
        { type: 'line', x0: 1, x1: maxDay, y0: p.ss, y1: p.ss,
          line: { color: '#f39c12', width: 1.5, dash: 'dot' } }
    ];

    const annotations = [
        { x: maxDay, y: p.rop, text: 'ROP', showarrow: false, xanchor: 'right', font: { color: '#e74c3c', size: 10 } },
        { x: maxDay, y: p.ss, text: 'SS', showarrow: false, xanchor: 'right', font: { color: '#f39c12', size: 10 } }
    ];

    Plotly.react('ss-chart', [invTrace, stockoutTrace, arrivalTrace, demandTrace], {
        margin: { t: 10, b: 40, l: 50, r: 50 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a', size: 10 },
        xaxis: { title: 'Day', gridcolor: 'rgba(255,255,255,0.07)', range: [1, maxDay] },
        yaxis: { title: 'Inventory (units)', gridcolor: 'rgba(255,255,255,0.07)', rangemode: 'tozero' },
        yaxis2: { title: 'Demand', overlaying: 'y', side: 'right', showgrid: false, rangemode: 'tozero' },
        shapes, annotations,
        legend: { orientation: 'h', y: -0.22, font: { size: 10 } },
        showlegend: true
    }, { responsive: true, displayModeBar: false });
}

function ssUpdateMetrics() {
    const s = ssSimState;
    const day = s.day;
    document.getElementById('ss-day-counter').textContent = day;
    document.getElementById('ss-m-inventory').innerHTML = Math.round(s.inventory) + '<span class="calc-result-unit"> units</span>';
    const sl = day > 0 ? ((day - s.stockoutDays) / day * 100) : 100;
    document.getElementById('ss-m-service').innerHTML = sl.toFixed(1) + '<span class="calc-result-unit">%</span>';
    document.getElementById('ss-m-stockouts').innerHTML = s.stockoutDays + '<span class="calc-result-unit"> days</span>';
    const avgInv = day > 0 ? s.history.inventory.reduce((a, b) => a + b, 0) / day : 0;
    document.getElementById('ss-m-avg-inv').innerHTML = Math.round(avgInv) + '<span class="calc-result-unit"> units</span>';
    document.getElementById('ss-m-orders').textContent = s.ordersPlaced;
    const fillRate = s.totalDemand > 0 ? (s.totalServed / s.totalDemand * 100) : 100;
    document.getElementById('ss-m-fillrate').innerHTML = fillRate.toFixed(1) + '<span class="calc-result-unit">%</span>';
}

function ssUpdateCosts() {
    const s = ssSimState;
    document.getElementById('ss-cost-holding').textContent = '$' + Math.round(s.totalHoldCost).toLocaleString();
    document.getElementById('ss-cost-stockout').textContent = '$' + Math.round(s.totalStockoutCost).toLocaleString();
    document.getElementById('ss-cost-total').textContent = '$' + Math.round(s.totalHoldCost + s.totalStockoutCost).toLocaleString();

    Plotly.react('ss-cost-chart', [{
        x: [Math.round(s.totalHoldCost), Math.round(s.totalStockoutCost)],
        y: ['Holding', 'Stockout'],
        type: 'bar', orientation: 'h',
        marker: { color: ['#27ae60', '#e74c3c'] },
        text: ['$' + Math.round(s.totalHoldCost).toLocaleString(), '$' + Math.round(s.totalStockoutCost).toLocaleString()],
        textposition: 'auto', textfont: { color: '#fff', size: 11 }
    }], {
        margin: { t: 5, b: 20, l: 70, r: 20 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a', size: 10 },
        xaxis: { gridcolor: 'rgba(255,255,255,0.07)' },
        yaxis: { automargin: true },
        showlegend: false, bargap: 0.3
    }, { responsive: true, displayModeBar: false });
}

function ssUpdateStockoutLog() {
    const log = ssSimState.stockoutLog;
    if (log.length === 0) {
        document.getElementById('ss-stockout-log').innerHTML = '<em>No stockouts yet.</em>';
        return;
    }
    const rows = log.slice(-20).map(e => {
        const severity = e.shortfall > e.demand * 0.5 ? '#e74c3c' : (e.shortfall > e.demand * 0.1 ? '#f39c12' : 'var(--text-dim)');
        return `<tr>
            <td style="padding:4px 8px;">Day ${e.day}</td>
            <td style="padding:4px 8px;">${e.demand}</td>
            <td style="padding:4px 8px;">${e.inventoryBefore}</td>
            <td style="padding:4px 8px; color:${severity}; font-weight:600;">${e.shortfall}</td>
        </tr>`;
    }).join('');
    document.getElementById('ss-stockout-log').innerHTML = `
        <table style="width:100%; border-collapse:collapse;">
            <tr style="color:var(--text-dim); font-size:11px; text-transform:uppercase; border-bottom:1px solid var(--border);">
                <th style="padding:4px 8px; text-align:left;">Day</th>
                <th style="padding:4px 8px; text-align:left;">Demand</th>
                <th style="padding:4px 8px; text-align:left;">Inv Before</th>
                <th style="padding:4px 8px; text-align:left;">Shortfall</th>
            </tr>
            ${rows}
        </table>
        ${log.length > 20 ? '<div style="margin-top:8px; font-size:11px; color:var(--text-dim);">Showing last 20 of ' + log.length + ' stockout events</div>' : ''}`;
}

function ssUpdateInsights() {
    const s = ssSimState;
    const p = s.params;
    const day = s.day;
    if (day < 5) return;

    const zToSL = { '1.28': 90, '1.65': 95, '1.96': 97.5, '2.33': 99, '2.58': 99.5 };
    const targetSL = zToSL[p.z.toFixed(2)] || 95;
    const actualSL = day > 0 ? ((day - s.stockoutDays) / day * 100) : 100;
    const avgInv = s.history.inventory.reduce((a, b) => a + b, 0) / day;
    const fillRate = s.totalDemand > 0 ? (s.totalServed / s.totalDemand * 100) : 100;

    let html = '<div style="line-height:1.7; color:var(--text-secondary);">';
    const gap = actualSL - targetSL;

    if (gap < -3) {
        html += `<div style="color:#e74c3c; font-weight:600; margin-bottom:8px;">
            &#9888; Underperforming: actual service level ${actualSL.toFixed(1)}% vs target ${targetSL}%</div>`;
        html += `<div style="margin-bottom:6px;">Safety stock appears insufficient for this variability.
            Stockouts on ${s.stockoutDays} of ${day} days.</div>`;
        html += `<div>Try: increase service level target, raise the SS multiplier, or reduce demand variability.</div>`;
    } else if (gap > 5) {
        html += `<div style="color:#27ae60; font-weight:600; margin-bottom:8px;">
            &#9989; Over-provisioned: actual ${actualSL.toFixed(1)}% vs target ${targetSL}%</div>`;
        html += `<div style="margin-bottom:6px;">Average inventory is ${Math.round(avgInv)} units &mdash;
            holding cost may be higher than necessary.</div>`;
        html += `<div>Consider reducing safety stock (lower SS multiplier or service level) to cut costs.</div>`;
    } else {
        html += `<div style="color:#27ae60; font-weight:600; margin-bottom:8px;">
            &#9989; On target: actual ${actualSL.toFixed(1)}% &asymp; target ${targetSL}%</div>`;
        html += `<div>Policy is well-calibrated. ${s.stockoutDays} stockout day(s), fill rate ${fillRate.toFixed(1)}%.</div>`;
    }

    html += `<ul style="margin-top:12px; padding-left:20px; font-size:12px; color:var(--text-dim);">
        <li>Avg inventory: ${Math.round(avgInv)} units &times; $${p.holdCost}/day = $${Math.round(avgInv * p.holdCost)}/day holding</li>
        <li>Orders placed: ${s.ordersPlaced} (avg cycle: ${s.ordersPlaced > 0 ? Math.round(day / s.ordersPlaced) : '—'} days)</li>
        <li>Fill rate: ${fillRate.toFixed(1)}% of demand units served from stock</li>
    </ul></div>`;

    document.getElementById('ss-insights').innerHTML = html;
}

function ssUpdateMultiplier(val) {
    const mult = parseInt(val) / 100;
    document.getElementById('ss-multiplier-label').textContent = mult.toFixed(1) + '\u00d7';
    const d = ssComputeDerived();
    const effSS = Math.round(d.ss * mult);
    const effROP = Math.round(d.demand * d.lead + effSS);
    document.getElementById('ss-wif-ss').textContent = effSS + ' units';
    document.getElementById('ss-wif-rop').textContent = effROP + ' units';
    // Estimate cost impact vs baseline
    const baseSS = Math.round(d.ss);
    const delta = effSS - baseSS;
    const holdCost = parseFloat(document.getElementById('ss-hold-cost').value) || 0;
    const duration = parseInt(document.getElementById('ss-duration').value) || 180;
    const costDelta = delta * holdCost * duration;
    if (delta === 0) {
        document.getElementById('ss-wif-cost').textContent = 'baseline';
    } else {
        const sign = costDelta > 0 ? '+' : '';
        document.getElementById('ss-wif-cost').textContent = sign + '$' + Math.round(Math.abs(costDelta)).toLocaleString() + (costDelta > 0 ? ' holding' : ' saved');
        document.getElementById('ss-wif-cost').style.color = costDelta > 0 ? '#e74c3c' : '#27ae60';
    }
}

function ssStop() {
    clearInterval(ssSimState.interval);
    ssSimState.running = false;
    document.getElementById('ss-start-btn').innerHTML = '&#9654; Replay';
    // Final updates
    ssUpdateChart();
    ssUpdateMetrics();
    ssUpdateCosts();
    ssUpdateStockoutLog();
    ssUpdateInsights();
    // Publish to SvendOps
    const day = ssSimState.day;
    const sl = day > 0 ? ((day - ssSimState.stockoutDays) / day * 100) : 100;
    const avgInv = day > 0 ? ssSimState.history.inventory.reduce((a, b) => a + b, 0) / day : 0;
    SvendOps.publish('safetySimServiceLevel', sl.toFixed(1), '%', 'SS Sim');
    SvendOps.publish('safetySimAvgInventory', Math.round(avgInv), 'units', 'SS Sim');
    SvendOps.publish('safetySimStockouts', ssSimState.stockoutDays, 'days', 'SS Sim');
}

function ssPullFromSafety() {
    const fields = [
        ['safety-demand', 'ss-demand'],
        ['safety-demand-std', 'ss-demand-std'],
        ['safety-lead', 'ss-lead'],
        ['safety-lead-std', 'ss-lead-std'],
        ['safety-service', 'ss-service']
    ];
    let pulled = 0;
    fields.forEach(([src, dst]) => {
        const srcEl = document.getElementById(src);
        const dstEl = document.getElementById(dst);
        if (srcEl && dstEl && srcEl.value) {
            dstEl.value = srcEl.value;
            dstEl.style.transition = 'background 0.3s';
            dstEl.style.background = 'rgba(74,159,110,0.2)';
            setTimeout(() => dstEl.style.background = '', 600);
            pulled++;
        }
    });
    if (pulled > 0) ssUpdateDerived();
}

function ssPullFromEOQ() {
    const eoqVal = SvendOps.get('eoq');
    if (eoqVal) {
        const el = document.getElementById('ss-order-qty');
        el.value = Math.round(eoqVal);
        el.style.transition = 'background 0.3s';
        el.style.background = 'rgba(74,159,110,0.2)';
        setTimeout(() => el.style.background = '', 600);
    }
}

function loadVSMIntoSafetySim(stations) {
    if (!stations || stations.length === 0) return;
    // Use first station's cycle time to derive daily demand
    // Assume 480 min/day (8-hour shift): demand = 480*60 / avg_ct
    const avgCT = stations.reduce((s, st) => s + (st.ct || 60), 0) / stations.length;
    const dailyDemand = Math.round((480 * 60) / avgCT);
    document.getElementById('ss-demand').value = dailyDemand;
    ssUpdateDerived();
}

// ============================================================================
// FMEA Monte Carlo — Risk Distribution Simulation
// ============================================================================

let fmeaSimItems = [
    { mode: 'Seal failure', severity: 8, occurrence: 4, detection: 5 },
    { mode: 'Wrong label', severity: 5, occurrence: 3, detection: 2 },
    { mode: 'Underfill', severity: 7, occurrence: 2, detection: 3 },
    { mode: 'Contamination', severity: 9, occurrence: 2, detection: 4 },
    { mode: 'Cracked housing', severity: 6, occurrence: 3, detection: 6 },
];

function fmeaSimRenderItems() {
    const c = document.getElementById('fmea-sim-items');
    if (!c) return;
    c.innerHTML = fmeaSimItems.map((f, i) => {
        const rpn = f.severity * f.occurrence * f.detection;
        const rpnColor = rpn > 100 ? '#e74c3c' : rpn > 50 ? '#f39c12' : '#4a9f6e';
        return `
        <div style="display:grid; grid-template-columns:1fr 80px 80px 80px 60px 30px; gap:4px; align-items:center; padding:8px 12px; background:var(--bg-secondary); border-radius:8px; margin-bottom:3px;">
            <input type="text" value="${f.mode}" placeholder="Failure Mode" style="padding:6px 8px; font-size:13px;" oninput="fmeaSimUpdate(${i},'mode',this.value)">
            <input type="number" value="${f.severity}" min="1" max="10" style="text-align:center; padding:6px;" oninput="fmeaSimUpdate(${i},'severity',this.value)" title="Severity">
            <input type="number" value="${f.occurrence}" min="1" max="10" style="text-align:center; padding:6px;" oninput="fmeaSimUpdate(${i},'occurrence',this.value)" title="Occurrence">
            <input type="number" value="${f.detection}" min="1" max="10" style="text-align:center; padding:6px;" oninput="fmeaSimUpdate(${i},'detection',this.value)" title="Detection">
            <div style="text-align:center; font-weight:700; color:${rpnColor}; font-size:13px;">${rpn}</div>
            <button class="yamazumi-station-remove" onclick="fmeaSimRemove(${i})">&times;</button>
        </div>`;
    }).join('');
}

function fmeaSimUpdate(idx, field, value) {
    if (field !== 'mode') value = Math.min(10, Math.max(1, parseInt(value) || 1));
    fmeaSimItems[idx][field] = value;
    fmeaSimRenderItems();
}

function fmeaSimRemove(idx) {
    if (fmeaSimItems.length <= 1) return;
    fmeaSimItems.splice(idx, 1);
    fmeaSimRenderItems();
}

function fmeaSimAddItem() {
    fmeaSimItems.push({ mode: 'New failure mode', severity: 5, occurrence: 5, detection: 5 });
    fmeaSimRenderItems();
}

function fmeaSimLoadExample() {
    fmeaSimItems = [
        { mode: 'Seal failure', severity: 8, occurrence: 4, detection: 5 },
        { mode: 'Wrong label', severity: 5, occurrence: 3, detection: 2 },
        { mode: 'Underfill', severity: 7, occurrence: 2, detection: 3 },
        { mode: 'Contamination', severity: 9, occurrence: 2, detection: 4 },
        { mode: 'Cracked housing', severity: 6, occurrence: 3, detection: 6 },
    ];
    fmeaSimRenderItems();
}

function fmeaSimPullFromFMEA() {
    if (typeof fmeaData !== 'undefined' && fmeaData.length > 0) {
        fmeaSimItems = fmeaData.map(f => ({
            mode: f.mode, severity: f.severity, occurrence: f.occurrence, detection: f.detection
        }));
        fmeaSimRenderItems();
    }
}

function fmeaSimTriangular(min, mode, max) {
    // Sample from triangular distribution
    const u = Math.random();
    const fc = (mode - min) / (max - min);
    if (u < fc) return min + Math.sqrt(u * (max - min) * (mode - min));
    return max - Math.sqrt((1 - u) * (max - min) * (max - mode));
}

function fmeaSimRun() {
    const runs = parseInt(document.getElementById('fmea-sim-runs').value) || 2000;
    const unc = parseFloat(document.getElementById('fmea-sim-uncertainty').value) || 1;
    const threshold = parseInt(document.getElementById('fmea-sim-threshold').value) || 100;

    const btn = document.getElementById('fmea-sim-run-btn');
    btn.textContent = 'Running...';
    btn.disabled = true;

    // Use setTimeout to allow UI to update
    setTimeout(() => {
        // Per-mode RPN distributions
        const modeRPNs = fmeaSimItems.map(() => []);
        const systemRPNs = [];
        let exceedCount = 0;

        for (let r = 0; r < runs; r++) {
            let systemRPN = 0;
            let anyExceed = false;

            fmeaSimItems.forEach((f, i) => {
                // Sample S, O, D from triangular distributions
                const sMin = Math.max(1, f.severity - unc);
                const sMax = Math.min(10, f.severity + unc);
                const oMin = Math.max(1, f.occurrence - unc);
                const oMax = Math.min(10, f.occurrence + unc);
                const dMin = Math.max(1, f.detection - unc);
                const dMax = Math.min(10, f.detection + unc);

                const s = fmeaSimTriangular(sMin, f.severity, sMax);
                const o = fmeaSimTriangular(oMin, f.occurrence, oMax);
                const d = fmeaSimTriangular(dMin, f.detection, dMax);

                const rpn = s * o * d;
                modeRPNs[i].push(rpn);
                systemRPN += rpn;
                if (rpn > threshold) anyExceed = true;
            });

            systemRPNs.push(systemRPN);
            if (anyExceed) exceedCount++;
        }

        // Compute statistics
        const mean = systemRPNs.reduce((a, b) => a + b, 0) / runs;
        const max = Math.max(...systemRPNs);
        const sorted = [...systemRPNs].sort((a, b) => a - b);
        const p5 = sorted[Math.floor(runs * 0.05)];
        const p95 = sorted[Math.floor(runs * 0.95)];
        const pExceed = (exceedCount / runs * 100);

        // Per-mode stats
        const modeStats = fmeaSimItems.map((f, i) => {
            const rpns = modeRPNs[i];
            const modeMean = rpns.reduce((a, b) => a + b, 0) / runs;
            const modeMax = Math.max(...rpns);
            const modeSorted = [...rpns].sort((a, b) => a - b);
            const modeP95 = modeSorted[Math.floor(runs * 0.95)];
            const modeExceed = rpns.filter(r => r > threshold).length / runs * 100;
            return { name: f.mode, mean: modeMean, max: modeMax, p95: modeP95, pExceed: modeExceed, rpns };
        });

        // Find worst mode by contribution to mean
        const worstMode = modeStats.reduce((a, b) => a.mean > b.mean ? a : b);

        // Update results
        document.getElementById('fmea-sim-results').style.display = '';
        document.getElementById('fmea-sim-p-exceed').innerHTML = `${pExceed.toFixed(1)}<span class="calc-result-unit">%</span>`;
        document.getElementById('fmea-sim-avg-rpn').textContent = Math.round(mean);
        document.getElementById('fmea-sim-max-rpn').textContent = Math.round(max);
        document.getElementById('fmea-sim-worst-mode').textContent = worstMode.name;

        // System RPN histogram
        document.getElementById('fmea-sim-chart-area').style.display = '';
        Plotly.newPlot('fmea-sim-chart', [{
            x: systemRPNs,
            type: 'histogram',
            nbinsx: 50,
            marker: { color: 'rgba(74,159,110,0.7)', line: { color: '#4a9f6e', width: 1 } },
            name: 'System RPN'
        }], {
            shapes: [{
                type: 'line', x0: threshold * fmeaSimItems.length, x1: threshold * fmeaSimItems.length,
                y0: 0, y1: 1, yref: 'paper',
                line: { color: '#e74c3c', width: 2, dash: 'dash' }
            }],
            annotations: [{
                x: threshold * fmeaSimItems.length, y: 1, yref: 'paper',
                text: `Threshold × ${fmeaSimItems.length}`,
                showarrow: false, font: { color: '#e74c3c', size: 10 }, yshift: 10
            }],
            margin: { t: 20, b: 50, l: 50, r: 20 },
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#9aaa9a' },
            xaxis: { title: 'System RPN (sum of all modes)', gridcolor: 'rgba(255,255,255,0.1)' },
            yaxis: { title: 'Frequency', gridcolor: 'rgba(255,255,255,0.1)' },
        }, { responsive: true, displayModeBar: false });

        // Per-mode box plot
        document.getElementById('fmea-sim-mode-chart-area').style.display = '';
        const modeTraces = modeStats.map((ms, i) => ({
            y: ms.rpns.filter((_, j) => j % Math.max(1, Math.floor(runs / 500)) === 0), // sample for perf
            type: 'box',
            name: ms.name,
            marker: { color: `hsl(${(i * 137) % 360}, 60%, 50%)` },
            boxpoints: false,
        }));
        Plotly.newPlot('fmea-sim-mode-chart', modeTraces, {
            margin: { t: 20, b: 80, l: 50, r: 20 },
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#9aaa9a' },
            yaxis: { title: 'RPN', gridcolor: 'rgba(255,255,255,0.1)' },
            xaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
            shapes: [{
                type: 'line', x0: -0.5, x1: modeStats.length - 0.5,
                y0: threshold, y1: threshold,
                line: { color: '#e74c3c', width: 2, dash: 'dash' }
            }],
            showlegend: false,
        }, { responsive: true, displayModeBar: false });

        // Tornado chart (risk contribution)
        document.getElementById('fmea-sim-tornado-area').style.display = '';
        const sortedModes = [...modeStats].sort((a, b) => b.mean - a.mean);
        Plotly.newPlot('fmea-sim-tornado', [{
            y: sortedModes.map(m => m.name),
            x: sortedModes.map(m => m.mean),
            type: 'bar',
            orientation: 'h',
            marker: { color: sortedModes.map(m => m.pExceed > 20 ? '#e74c3c' : m.pExceed > 5 ? '#f39c12' : '#4a9f6e') },
            text: sortedModes.map(m => `Avg: ${m.mean.toFixed(0)}, P(>${threshold}): ${m.pExceed.toFixed(0)}%`),
            textposition: 'outside',
            textfont: { size: 10 },
        }], {
            margin: { t: 20, b: 50, l: 120, r: 100 },
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#9aaa9a' },
            xaxis: { title: 'Average RPN', gridcolor: 'rgba(255,255,255,0.1)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.1)', autorange: 'reversed' },
            shapes: [{
                type: 'line', x0: threshold, x1: threshold,
                y0: -0.5, y1: sortedModes.length - 0.5,
                line: { color: '#e74c3c', width: 2, dash: 'dash' }
            }],
        }, { responsive: true, displayModeBar: false });

        // Insights
        fmeaSimUpdateInsights(runs, threshold, pExceed, mean, max, p5, p95, modeStats, worstMode);

        // Publish
        SvendOps.publish('fmeaSimPExceed', parseFloat(pExceed.toFixed(1)), '%', 'FMEA MC');
        SvendOps.publish('fmeaSimAvgRPN', Math.round(mean), '', 'FMEA MC');

        btn.textContent = 'Run Monte Carlo Simulation';
        btn.disabled = false;
    }, 50);
}

function fmeaSimUpdateInsights(runs, threshold, pExceed, mean, max, p5, p95, modeStats, worstMode) {
    let html = `<div style="margin-bottom:10px;"><strong>${runs.toLocaleString()} simulations complete.</strong> System RPN range: ${Math.round(p5)} (5th pctl) to ${Math.round(p95)} (95th pctl).</div>`;

    // Compounding risk highlight
    const modeSafe = modeStats.filter(m => m.pExceed < 10);
    const modeDangerous = modeStats.filter(m => m.pExceed >= 10);

    if (pExceed > 50) {
        html += `<div style="padding:10px 12px; background:rgba(231,76,60,0.15); border-radius:6px; color:#e74c3c; margin-bottom:10px; font-weight:600;">
            High risk: ${pExceed.toFixed(0)}% of scenarios have at least one failure mode exceeding RPN ${threshold}. Immediate action required on ${modeDangerous.length} mode${modeDangerous.length > 1 ? 's' : ''}.
        </div>`;
    } else if (pExceed > 20) {
        html += `<div style="padding:10px 12px; background:rgba(243,156,18,0.15); border-radius:6px; color:#f39c12; margin-bottom:10px;">
            Moderate risk: ${pExceed.toFixed(0)}% chance of threshold exceedance. Focus mitigation on "${worstMode.name}" which contributes ${(worstMode.mean / mean * 100).toFixed(0)}% of system risk.
        </div>`;
    } else {
        html += `<div style="padding:10px 12px; background:rgba(74,159,110,0.15); border-radius:6px; color:var(--accent); margin-bottom:10px;">
            Low risk: Only ${pExceed.toFixed(1)}% of scenarios exceed threshold. Current controls appear adequate.
        </div>`;
    }

    // Per-mode breakdown
    html += '<div style="margin-top:10px; font-size:12px;">';
    html += '<strong>Per-mode exceedance probability:</strong><br>';
    [...modeStats].sort((a, b) => b.pExceed - a.pExceed).forEach(m => {
        const bar = Math.min(100, m.pExceed);
        const color = m.pExceed > 20 ? '#e74c3c' : m.pExceed > 5 ? '#f39c12' : '#4a9f6e';
        html += `<div style="display:flex; align-items:center; gap:8px; margin-top:4px;">
            <span style="min-width:120px; font-size:11px;">${m.name}</span>
            <div style="flex:1; height:12px; background:var(--bg-tertiary); border-radius:3px; overflow:hidden;">
                <div style="width:${bar}%; height:100%; background:${color}; border-radius:3px;"></div>
            </div>
            <span style="font-size:11px; min-width:50px; text-align:right; color:${color}; font-weight:600;">${m.pExceed.toFixed(1)}%</span>
        </div>`;
    });
    html += '</div>';

    document.getElementById('fmea-sim-insights').innerHTML = html;

    renderNextSteps('fmea-sim-next-steps', [
        { title: 'FMEA / RPN', desc: 'Detailed failure mode analysis and scoring', calcId: 'fmea' },
        { title: 'Risk Matrix', desc: 'Visual risk assessment for high-RPN modes', calcId: 'riskmatrix' },
        { title: 'Cp / Cpk', desc: 'Check process capability for critical modes', calcId: 'cpk' },
    ]);
    document.getElementById('fmea-sim-next-steps').style.display = '';
}

// Init FMEA MC on load
document.addEventListener('DOMContentLoaded', () => { fmeaSimRenderItems(); });

// ============================================================================
// SMED Simulator — Interactive Changeover Reduction
// ============================================================================

let smedSimElements = [
    { name: 'Gather tools & parts', time: 8, type: 'internal', convertible: true },
    { name: 'Remove old die/fixture', time: 12, type: 'internal', convertible: false },
    { name: 'Clean mounting surface', time: 5, type: 'internal', convertible: true },
    { name: 'Install new die/fixture', time: 15, type: 'internal', convertible: false },
    { name: 'Align & adjust settings', time: 10, type: 'internal', convertible: false },
    { name: 'Connect utilities', time: 6, type: 'internal', convertible: true },
    { name: 'Preheat new die', time: 10, type: 'internal', convertible: true },
    { name: 'First piece inspection', time: 5, type: 'internal', convertible: false },
    { name: 'Cleanup & store old die', time: 8, type: 'internal', convertible: true },
];

let smedSimAnim = { running: false, interval: null, time: 0, phase: null };

function smedSimRenderElements() {
    const c = document.getElementById('smed-sim-elements');
    if (!c) return;
    c.innerHTML = smedSimElements.map((el, i) => {
        const isInt = el.type === 'internal';
        const btnColor = isInt ? '#e74c3c' : '#4a9f6e';
        const btnText = isInt ? 'Internal' : 'External';
        return `
        <div style="display:flex; align-items:center; gap:8px; padding:10px 12px; background:var(--bg-secondary); border-radius:8px; margin-bottom:4px; border-left:3px solid ${btnColor};">
            <button onclick="smedSimToggleType(${i})" style="padding:4px 12px; background:${btnColor}; color:white; border:none; border-radius:4px; cursor:pointer; font-size:11px; font-weight:600; min-width:70px;">${btnText}</button>
            <input type="text" value="${el.name}" style="flex:1; padding:6px 8px; font-size:13px;" oninput="smedSimUpdate(${i},'name',this.value)">
            <input type="number" value="${el.time}" min="1" max="120" style="width:60px; text-align:right; padding:6px;" oninput="smedSimUpdate(${i},'time',this.value)">
            <span style="color:var(--text-dim); font-size:11px;">min</span>
            <button class="yamazumi-station-remove" onclick="smedSimRemove(${i})">&times;</button>
        </div>`;
    }).join('');
    smedSimCalc();
}

function smedSimToggleType(idx) {
    smedSimElements[idx].type = smedSimElements[idx].type === 'internal' ? 'external' : 'internal';
    smedSimRenderElements();
}

function smedSimUpdate(idx, field, value) {
    if (field === 'time') value = Math.max(1, parseInt(value) || 1);
    smedSimElements[idx][field] = value;
    smedSimCalc();
}

function smedSimRemove(idx) {
    smedSimElements.splice(idx, 1);
    smedSimRenderElements();
}

function smedSimAddElement() {
    smedSimElements.push({ name: 'New element', time: 5, type: 'internal', convertible: true });
    smedSimRenderElements();
}

function smedSimLoadExample() {
    smedSimElements = [
        { name: 'Gather tools & parts', time: 8, type: 'internal', convertible: true },
        { name: 'Remove old die/fixture', time: 12, type: 'internal', convertible: false },
        { name: 'Clean mounting surface', time: 5, type: 'internal', convertible: true },
        { name: 'Install new die/fixture', time: 15, type: 'internal', convertible: false },
        { name: 'Align & adjust settings', time: 10, type: 'internal', convertible: false },
        { name: 'Connect utilities', time: 6, type: 'internal', convertible: true },
        { name: 'Preheat new die', time: 10, type: 'internal', convertible: true },
        { name: 'First piece inspection', time: 5, type: 'internal', convertible: false },
        { name: 'Cleanup & store old die', time: 8, type: 'internal', convertible: true },
    ];
    smedSimRenderElements();
}

function smedSimPullFromSMED() {
    if (typeof smedData !== 'undefined' && smedData.length > 0) {
        smedSimElements = smedData.map(s => ({
            name: s.name, time: s.time, type: s.type, convertible: true
        }));
        smedSimRenderElements();
    }
}

function smedSimCalc() {
    const internal = smedSimElements.filter(e => e.type === 'internal');
    const external = smedSimElements.filter(e => e.type === 'external');
    const totalBefore = smedSimElements.reduce((s, e) => s + e.time, 0);
    const internalTime = internal.reduce((s, e) => s + e.time, 0);
    const externalTime = external.reduce((s, e) => s + e.time, 0);
    const saved = totalBefore - internalTime;
    const reduction = totalBefore > 0 ? ((saved / totalBefore) * 100) : 0;

    // Update results
    document.getElementById('smed-sim-total-before').innerHTML = `${totalBefore}<span class="calc-result-unit">min</span>`;
    document.getElementById('smed-sim-total-after').innerHTML = `${internalTime}<span class="calc-result-unit">min</span>`;
    document.getElementById('smed-sim-saved').innerHTML = `${saved}<span class="calc-result-unit">min</span>`;
    document.getElementById('smed-sim-reduction').innerHTML = `${reduction.toFixed(0)}<span class="calc-result-unit">%</span>`;

    // Update timeline headers
    document.getElementById('smed-sim-before-time').textContent = `(${totalBefore} min total)`;
    document.getElementById('smed-sim-after-time').textContent = `(${internalTime} min downtime)`;

    // Render before bar (all sequential)
    const maxTime = Math.max(totalBefore, 1);
    let beforeHtml = '';
    smedSimElements.forEach(el => {
        const w = (el.time / maxTime * 100).toFixed(1);
        const color = el.type === 'internal' ? '#e74c3c' : '#c0392b';
        beforeHtml += `<div style="width:${w}%; min-width:20px; height:24px; background:${color}; border-radius:3px; display:flex; align-items:center; justify-content:center; font-size:9px; color:white; overflow:hidden; white-space:nowrap; padding:0 4px;" title="${el.name} (${el.time} min)">${el.time > 3 ? el.name.substring(0, 12) : el.time}</div>`;
    });
    document.getElementById('smed-sim-before-bar').innerHTML = beforeHtml;

    // Render after bars (separated)
    let extHtml = '', intHtml = '';
    external.forEach(el => {
        const w = (el.time / maxTime * 100).toFixed(1);
        extHtml += `<div style="width:${w}%; min-width:20px; height:24px; background:#4a9f6e; border-radius:3px; display:flex; align-items:center; justify-content:center; font-size:9px; color:white; overflow:hidden; white-space:nowrap; padding:0 4px;" title="${el.name} (${el.time} min)">${el.time > 3 ? el.name.substring(0, 12) : el.time}</div>`;
    });
    internal.forEach(el => {
        const w = (el.time / maxTime * 100).toFixed(1);
        intHtml += `<div style="width:${w}%; min-width:20px; height:24px; background:#e74c3c; border-radius:3px; display:flex; align-items:center; justify-content:center; font-size:9px; color:white; overflow:hidden; white-space:nowrap; padding:0 4px;" title="${el.name} (${el.time} min)">${el.time > 3 ? el.name.substring(0, 12) : el.time}</div>`;
    });
    document.getElementById('smed-sim-ext-bar').innerHTML = extHtml || '<div style="font-size:11px; color:var(--text-dim);">No external elements yet — click Internal buttons above to convert</div>';
    document.getElementById('smed-sim-int-bar').innerHTML = intHtml || '<div style="font-size:11px; color:var(--text-dim);">All elements externalized!</div>';

    // Update chart
    smedSimUpdateChart(totalBefore, internalTime, externalTime, internal, external);

    // Update insights
    smedSimUpdateInsights(totalBefore, internalTime, externalTime, reduction, internal, external);

    // Publish
    SvendOps.publish('smedSimBefore', totalBefore, 'min', 'SMED Sim');
    SvendOps.publish('smedSimAfter', internalTime, 'min', 'SMED Sim');
}

function smedSimUpdateChart(totalBefore, internalTime, externalTime, internal, external) {
    // Gantt-style comparison chart
    const beforeTasks = [], afterIntTasks = [], afterExtTasks = [];
    let bCum = 0;
    smedSimElements.forEach(el => {
        beforeTasks.push({ name: el.name, start: bCum, duration: el.time, type: 'before' });
        bCum += el.time;
    });
    let iCum = 0;
    internal.forEach(el => {
        afterIntTasks.push({ name: el.name, start: iCum, duration: el.time, type: 'internal' });
        iCum += el.time;
    });
    let eCum = 0;
    external.forEach(el => {
        afterExtTasks.push({ name: el.name, start: eCum, duration: el.time, type: 'external' });
        eCum += el.time;
    });

    const traces = [
        // Before row
        ...beforeTasks.map(t => ({
            x: [t.duration], y: ['Before SMED'], type: 'bar', orientation: 'h',
            base: [t.start], marker: { color: '#e74c3c' },
            text: [`${t.name} (${t.duration}m)`], textposition: 'inside',
            textfont: { size: 10, color: 'white' },
            hoverinfo: 'text', showlegend: false,
        })),
        // After internal row
        ...afterIntTasks.map(t => ({
            x: [t.duration], y: ['After (Internal)'], type: 'bar', orientation: 'h',
            base: [t.start], marker: { color: '#e74c3c' },
            text: [`${t.name} (${t.duration}m)`], textposition: 'inside',
            textfont: { size: 10, color: 'white' },
            hoverinfo: 'text', showlegend: false,
        })),
        // After external row
        ...afterExtTasks.map(t => ({
            x: [t.duration], y: ['After (External)'], type: 'bar', orientation: 'h',
            base: [t.start], marker: { color: '#4a9f6e' },
            text: [`${t.name} (${t.duration}m)`], textposition: 'inside',
            textfont: { size: 10, color: 'white' },
            hoverinfo: 'text', showlegend: false,
        })),
    ];

    Plotly.newPlot('smed-sim-chart', traces, {
        barmode: 'overlay',
        margin: { t: 20, b: 50, l: 120, r: 20 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a' },
        xaxis: { title: 'Time (minutes)', gridcolor: 'rgba(255,255,255,0.1)' },
        yaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
        shapes: [{
            type: 'line', x0: internalTime, x1: internalTime,
            y0: -0.5, y1: 2.5,
            line: { color: '#4a9f6e', width: 2, dash: 'dash' }
        }],
        annotations: internalTime < totalBefore ? [{
            x: internalTime, y: 2.5, text: `${internalTime}m downtime`,
            showarrow: false, font: { color: '#4a9f6e', size: 11 }, yshift: 10
        }] : [],
    }, { responsive: true, displayModeBar: false });
}

function smedSimUpdateInsights(totalBefore, internalTime, externalTime, reduction, internal, external) {
    let html = '';
    if (external.length === 0) {
        html = '<em>All elements are still internal. Click the "Internal" buttons to convert elements that can be done while the machine is running.</em>';
    } else {
        html += `<div style="margin-bottom:8px;"><strong>${reduction.toFixed(0)}% downtime reduction:</strong> ${totalBefore} min → ${internalTime} min machine downtime.</div>`;
        html += `<div style="margin-bottom:8px;">${external.length} element${external.length > 1 ? 's' : ''} moved to external (${externalTime} min done while machine runs).</div>`;

        if (internalTime <= 10) {
            html += `<div style="padding:8px 12px; background:rgba(74,159,110,0.15); border-radius:6px; color:var(--accent); font-weight:600;">
                Single-Minute Exchange achieved! Downtime is under 10 minutes.
            </div>`;
        } else if (internalTime <= 30) {
            html += `<div style="padding:8px 12px; background:rgba(243,156,18,0.15); border-radius:6px; color:#f39c12;">
                Good progress. ${internalTime - 10} more minutes of internal work to convert before reaching single-minute exchange.
            </div>`;
        }

        // Suggestion: largest remaining internal element
        if (internal.length > 0) {
            const largest = internal.reduce((a, b) => a.time > b.time ? a : b);
            html += `<div style="margin-top:8px; font-size:12px; color:var(--text-dim);">Largest remaining internal element: <strong>"${largest.name}" (${largest.time} min)</strong> — can any part of this be done externally?</div>`;
        }
    }
    document.getElementById('smed-sim-insights').innerHTML = html;

    renderNextSteps('smed-sim-next-steps', [
        { title: 'SMED Analysis', desc: 'Detailed changeover element analysis', calcId: 'smed' },
        { title: 'Heijunka Sim', desc: 'See how short changeovers enable leveled production', calcId: 'heijunka-sim' },
        { title: 'Line Simulator', desc: 'Test changeover impact on production line throughput', calcId: 'line-sim' },
    ]);
    document.getElementById('smed-sim-next-steps').style.display = '';
}

function smedSimAnimate() {
    if (smedSimAnim.running) {
        clearInterval(smedSimAnim.interval);
        smedSimAnim.running = false;
        document.getElementById('smed-sim-start').innerHTML = '&#9654; Resume';
        return;
    }

    const totalBefore = smedSimElements.reduce((s, e) => s + e.time, 0);
    const internalOnly = smedSimElements.filter(e => e.type === 'internal').reduce((s, e) => s + e.time, 0);

    if (smedSimAnim.time === 0) {
        document.getElementById('smed-sim-anim-area').style.display = 'block';
        smedSimAnim.totalBefore = totalBefore;
        smedSimAnim.totalAfter = internalOnly;
        smedSimAnim.beforeDone = false;
        smedSimAnim.afterDone = false;

        // Build element sequence for animation
        let bCum = 0;
        smedSimAnim.beforeSteps = smedSimElements.map(el => {
            const step = { name: el.name, start: bCum, end: bCum + el.time };
            bCum += el.time;
            return step;
        });
        let iCum = 0;
        smedSimAnim.afterSteps = smedSimElements.filter(e => e.type === 'internal').map(el => {
            const step = { name: el.name, start: iCum, end: iCum + el.time };
            iCum += el.time;
            return step;
        });
    }

    smedSimAnim.running = true;
    document.getElementById('smed-sim-start').innerHTML = '&#10074;&#10074; Pause';
    smedSimAnim.interval = setInterval(smedSimAnimTick, 100);
}

function smedSimAnimTick() {
    const speed = parseInt(document.getElementById('smed-sim-speed').value) || 5;
    smedSimAnim.time += speed * 0.5; // minutes per tick

    const t = smedSimAnim.time;
    const tb = smedSimAnim.totalBefore;
    const ta = smedSimAnim.totalAfter;

    // Update before side
    const bPct = Math.min(100, (t / tb) * 100);
    document.getElementById('smed-anim-before-progress').style.width = bPct + '%';
    const bMin = Math.min(t, tb);
    document.getElementById('smed-anim-before-clock').textContent = `${Math.floor(bMin)}:${String(Math.floor((bMin % 1) * 60)).padStart(2, '0')}`;
    const bStep = smedSimAnim.beforeSteps.find(s => t >= s.start && t < s.end);
    document.getElementById('smed-anim-before-label').textContent = bStep ? bStep.name : (t >= tb ? 'Complete' : '');

    // Update after side
    const aPct = Math.min(100, (t / ta) * 100);
    document.getElementById('smed-anim-after-progress').style.width = aPct + '%';
    const aMin = Math.min(t, ta);
    document.getElementById('smed-anim-after-clock').textContent = `${Math.floor(aMin)}:${String(Math.floor((aMin % 1) * 60)).padStart(2, '0')}`;
    const aStep = smedSimAnim.afterSteps.find(s => t >= s.start && t < s.end);
    if (t >= ta && !smedSimAnim.afterDone) {
        smedSimAnim.afterDone = true;
        document.getElementById('smed-anim-after-label').innerHTML = '<span style="color:var(--accent); font-weight:600;">Done! Machine restarted.</span>';
    } else if (!smedSimAnim.afterDone) {
        document.getElementById('smed-anim-after-label').textContent = aStep ? aStep.name : '';
    }

    if (t >= tb && !smedSimAnim.beforeDone) {
        smedSimAnim.beforeDone = true;
        document.getElementById('smed-anim-before-label').innerHTML = '<span style="color:#e74c3c;">Done. Machine restarting.</span>';
    }

    // Both done
    if (t >= tb) {
        clearInterval(smedSimAnim.interval);
        smedSimAnim.running = false;
        document.getElementById('smed-sim-start').innerHTML = '&#9654; Done';
    }
}

function smedSimResetAnim() {
    if (smedSimAnim.interval) clearInterval(smedSimAnim.interval);
    smedSimAnim = { running: false, interval: null, time: 0, phase: null };
    document.getElementById('smed-sim-start').innerHTML = '&#9654; Animate Changeover';
    document.getElementById('smed-sim-anim-area').style.display = 'none';
    document.getElementById('smed-anim-before-progress').style.width = '0%';
    document.getElementById('smed-anim-after-progress').style.width = '0%';
    document.getElementById('smed-anim-before-clock').textContent = '0:00';
    document.getElementById('smed-anim-after-clock').textContent = '0:00';
    document.getElementById('smed-anim-before-label').textContent = 'Waiting...';
    document.getElementById('smed-anim-after-label').textContent = 'Waiting...';
}

// Init SMED Sim on load
document.addEventListener('DOMContentLoaded', () => { smedSimRenderElements(); });

// ============================================================================
// Heijunka Simulator — Batched vs Leveled
// ============================================================================

let hjSimProducts = [
    { product: 'A', demand: 100, color: 'hsl(0, 60%, 50%)' },
    { product: 'B', demand: 60, color: 'hsl(137, 60%, 50%)' },
    { product: 'C', demand: 40, color: 'hsl(220, 60%, 50%)' },
];

const hjSim = {
    running: false, interval: null, time: 0,
    batched: null, leveled: null,
    history: { time: [], bWip: [], lWip: [] },
};

function hjSimProductColor(idx) {
    const hues = [0, 137, 220, 45, 280, 30, 180, 330];
    return `hsl(${hues[idx % hues.length]}, 60%, 50%)`;
}

function hjSimRenderProducts() {
    const c = document.getElementById('hj-sim-products');
    if (!c) return;
    c.innerHTML = hjSimProducts.map((p, i) => `
        <div class="yamazumi-station" style="gap:8px;">
            <div style="width:16px;height:16px;border-radius:4px;background:${p.color};flex-shrink:0;"></div>
            <input type="text" value="${p.product}" style="width:60px; padding:8px;" oninput="hjSimUpdateProduct(${i},'product',this.value)">
            <span style="color:var(--text-dim); font-size:12px;">Demand:</span>
            <input type="number" value="${p.demand}" style="width:80px; text-align:right; padding:8px;" oninput="hjSimUpdateProduct(${i},'demand',this.value)">
            <span style="color:var(--text-dim); font-size:11px;">units</span>
            <button class="yamazumi-station-remove" onclick="hjSimRemoveProduct(${i})">&times;</button>
        </div>
    `).join('');
}

function hjSimAddProduct() {
    const idx = hjSimProducts.length;
    hjSimProducts.push({
        product: String.fromCharCode(65 + idx),
        demand: 50,
        color: hjSimProductColor(idx)
    });
    hjSimRenderProducts();
}

function hjSimRemoveProduct(idx) {
    if (hjSimProducts.length <= 2) return;
    hjSimProducts.splice(idx, 1);
    hjSimProducts.forEach((p, i) => { p.color = hjSimProductColor(i); });
    hjSimRenderProducts();
}

function hjSimUpdateProduct(idx, field, value) {
    if (field === 'demand') value = Math.max(1, parseInt(value) || 1);
    hjSimProducts[idx][field] = value;
}

function hjSimPullFromHeijunka() {
    if (typeof heijunkaData !== 'undefined' && heijunkaData.length > 0) {
        hjSimProducts = heijunkaData.map((p, i) => ({
            product: p.product,
            demand: p.demand,
            color: hjSimProductColor(i)
        }));
        hjSimRenderProducts();
    }
}

function hjSimBuildBatchSequence(products) {
    // All of A, then all of B, then all of C
    const seq = [];
    const totalDemand = products.reduce((s, p) => s + p.demand, 0);
    const scale = Math.min(1, 200 / totalDemand); // cap total units for sim
    products.forEach(p => {
        const count = Math.max(1, Math.round(p.demand * scale));
        for (let i = 0; i < count; i++) seq.push(p.product);
    });
    return seq;
}

function hjSimBuildLeveledSequence(products) {
    // Smallest repeating pattern based on demand ratio
    const gcd = (a, b) => b === 0 ? a : gcd(b, a % b);
    const demands = products.map(p => Math.max(1, p.demand));
    const g = demands.reduce((a, b) => gcd(a, b));
    const pattern = [];
    products.forEach(p => {
        const count = Math.max(1, p.demand) / g;
        for (let i = 0; i < count; i++) pattern.push(p.product);
    });
    // Interleave: spread products evenly
    const interleaved = [];
    const counts = {};
    products.forEach(p => { counts[p.product] = Math.max(1, p.demand) / g; });
    const totalPattern = pattern.length;
    const remaining = { ...counts };
    for (let i = 0; i < totalPattern; i++) {
        let best = null, bestScore = -1;
        for (const p of products) {
            if (remaining[p.product] > 0) {
                const score = remaining[p.product] / counts[p.product];
                if (score > bestScore) { bestScore = score; best = p.product; }
            }
        }
        if (best) { interleaved.push(best); remaining[best]--; }
    }
    // Repeat pattern to match batched total
    const totalDemand = products.reduce((s, p) => s + p.demand, 0);
    const scale = Math.min(1, 200 / totalDemand);
    const totalUnits = Math.max(interleaved.length, Math.round(totalDemand * scale));
    const seq = [];
    for (let i = 0; i < totalUnits; i++) seq.push(interleaved[i % interleaved.length]);
    return seq;
}

function hjSimCreateLane(sequence, products) {
    const cycleTime = parseFloat(document.getElementById('hj-cycle').value) || 10;
    const changeover = parseFloat(document.getElementById('hj-changeover').value) || 30;
    const productMap = {};
    products.forEach(p => { productMap[p.product] = p; });

    return {
        sequence,
        idx: 0,              // current position in sequence
        producing: null,     // current product being made
        changeoverLeft: 0,   // seconds remaining in changeover
        cycleLeft: 0,        // seconds remaining in current unit
        wip: {},             // per-product WIP counts
        completed: {},       // per-product completed counts
        totalCompleted: 0,
        changeovers: 0,
        totalWip: 0,
        waitTimes: [],       // per-unit wait time
        productionLog: [],   // [{product, startTime, endTime}]
        cycleTime,
        changeover,
    };
}

function hjSimStart() {
    if (hjSim.running) {
        clearInterval(hjSim.interval);
        hjSim.running = false;
        document.getElementById('hj-start-btn').innerHTML = '&#9654; Resume';
        return;
    }
    if (hjSim.time === 0) {
        // Initialize both lanes
        const batchSeq = hjSimBuildBatchSequence(hjSimProducts);
        const levelSeq = hjSimBuildLeveledSequence(hjSimProducts);
        hjSim.batched = hjSimCreateLane(batchSeq, hjSimProducts);
        hjSim.leveled = hjSimCreateLane(levelSeq, hjSimProducts);
        hjSim.history = { time: [], bWip: [], lWip: [] };
        hjSimProducts.forEach(p => {
            hjSim.batched.wip[p.product] = 0;
            hjSim.batched.completed[p.product] = 0;
            hjSim.leveled.wip[p.product] = 0;
            hjSim.leveled.completed[p.product] = 0;
        });
    }
    hjSim.running = true;
    document.getElementById('hj-start-btn').innerHTML = '&#10074;&#10074; Pause';
    hjSim.interval = setInterval(hjSimTick, 100);
}

function hjSimReset() {
    if (hjSim.interval) clearInterval(hjSim.interval);
    hjSim.running = false;
    hjSim.time = 0;
    hjSim.batched = null;
    hjSim.leveled = null;
    hjSim.history = { time: [], bWip: [], lWip: [] };
    document.getElementById('hj-start-btn').innerHTML = '&#9654; Start';
    document.getElementById('hj-time-counter').textContent = '0';
    ['hj-b-wip','hj-b-changeovers','hj-b-completed'].forEach(id => document.getElementById(id).textContent = '0');
    ['hj-l-wip','hj-l-changeovers','hj-l-completed'].forEach(id => document.getElementById(id).textContent = '0');
    document.getElementById('hj-b-leadtime').textContent = '—';
    document.getElementById('hj-l-leadtime').textContent = '—';
    document.getElementById('hj-batched-visual').innerHTML = '<div style="color:var(--text-dim); font-size:12px; text-align:center;">Press Start</div>';
    document.getElementById('hj-leveled-visual').innerHTML = '<div style="color:var(--text-dim); font-size:12px; text-align:center;">Press Start</div>';
    document.getElementById('hj-sim-insights').innerHTML = '<em>Start simulation to compare batched vs leveled production...</em>';
    try { Plotly.purge('hj-sim-chart'); } catch(e) {}
}

function hjSimStepLane(lane, dt) {
    const demandInterval = parseFloat(document.getElementById('hj-demand-interval').value) || 15;
    for (let t = 0; t < dt; t++) {
        // If in changeover, count down
        if (lane.changeoverLeft > 0) {
            lane.changeoverLeft--;
            continue;
        }
        // If currently producing a unit, count down
        if (lane.cycleLeft > 0) {
            lane.cycleLeft--;
            if (lane.cycleLeft <= 0 && lane.producing) {
                // Unit completed
                lane.completed[lane.producing] = (lane.completed[lane.producing] || 0) + 1;
                lane.totalCompleted++;
                // Record wait time (simplified: position in sequence × cycle time + changeovers)
            }
            continue;
        }
        // Start next unit
        if (lane.idx < lane.sequence.length) {
            const nextProduct = lane.sequence[lane.idx];
            if (lane.producing && lane.producing !== nextProduct) {
                // Changeover needed
                lane.changeoverLeft = lane.changeover;
                lane.changeovers++;
                lane.producing = nextProduct;
                lane.idx++;
                lane.cycleLeft = lane.cycleTime;
                // Add WIP for all waiting units of this product type
                lane.wip[nextProduct] = (lane.wip[nextProduct] || 0) + 1;
            } else {
                lane.producing = nextProduct;
                lane.idx++;
                lane.cycleLeft = lane.cycleTime;
                lane.wip[nextProduct] = (lane.wip[nextProduct] || 0) + 1;
            }
        }
    }
    // Update total WIP (units started but not yet complete)
    const started = lane.idx;
    lane.totalWip = Math.max(0, started - lane.totalCompleted);
}

function hjSimTick() {
    const speed = parseInt(document.getElementById('hj-speed').value) || 5;
    const dt = speed;
    hjSim.time += dt;

    hjSimStepLane(hjSim.batched, dt);
    hjSimStepLane(hjSim.leveled, dt);

    // Record history every few ticks
    if (hjSim.time % 5 === 0) {
        hjSim.history.time.push(hjSim.time);
        hjSim.history.bWip.push(hjSim.batched.totalWip);
        hjSim.history.lWip.push(hjSim.leveled.totalWip);
    }

    // Stop when both lanes finish
    const bDone = hjSim.batched.idx >= hjSim.batched.sequence.length && hjSim.batched.cycleLeft <= 0 && hjSim.batched.changeoverLeft <= 0;
    const lDone = hjSim.leveled.idx >= hjSim.leveled.sequence.length && hjSim.leveled.cycleLeft <= 0 && hjSim.leveled.changeoverLeft <= 0;
    if (bDone && lDone) {
        clearInterval(hjSim.interval);
        hjSim.running = false;
        document.getElementById('hj-start-btn').innerHTML = '&#9654; Done';
        hjSimUpdateInsights();
    }

    hjSimUpdateMetrics();
    if (hjSim.time % 10 === 0) hjSimUpdateChart();
    hjSimUpdateVisuals();
}

function hjSimUpdateMetrics() {
    document.getElementById('hj-time-counter').textContent = hjSim.time;
    const b = hjSim.batched, l = hjSim.leveled;

    document.getElementById('hj-b-wip').textContent = b.totalWip;
    document.getElementById('hj-b-changeovers').textContent = b.changeovers;
    document.getElementById('hj-b-completed').textContent = b.totalCompleted;

    document.getElementById('hj-l-wip').textContent = l.totalWip;
    document.getElementById('hj-l-changeovers').textContent = l.changeovers;
    document.getElementById('hj-l-completed').textContent = l.totalCompleted;

    // Avg lead time estimate: (total time so far) / completed, weighted by WIP
    if (b.totalCompleted > 0) {
        const bAvgLT = ((b.totalWip * b.cycleTime / 2) + (b.changeovers * b.changeover / b.totalCompleted * (hjSimProducts.length - 1))).toFixed(0);
        document.getElementById('hj-b-leadtime').textContent = Math.round(hjSim.time / b.totalCompleted) + 's';
    }
    if (l.totalCompleted > 0) {
        document.getElementById('hj-l-leadtime').textContent = Math.round(hjSim.time / l.totalCompleted) + 's';
    }
}

function hjSimUpdateVisuals() {
    const productColors = {};
    hjSimProducts.forEach(p => { productColors[p.product] = p.color; });

    // Render production progress bars
    function renderLane(lane, containerId) {
        const el = document.getElementById(containerId);
        const total = lane.sequence.length;
        if (total === 0) return;

        // Show production queue as colored blocks
        const blockWidth = Math.max(2, Math.min(8, Math.floor(el.clientWidth / total)));
        let html = '<div style="display:flex; flex-wrap:wrap; gap:1px;">';
        for (let i = 0; i < total; i++) {
            const prod = lane.sequence[i];
            const color = productColors[prod] || '#666';
            const opacity = i < lane.idx ? '0.3' : i === lane.idx ? '1' : '0.6';
            const border = i === lane.idx ? '2px solid white' : 'none';
            html += `<div style="width:${blockWidth}px; height:18px; background:${color}; opacity:${opacity}; border:${border}; border-radius:2px;" title="${prod}"></div>`;
        }
        html += '</div>';

        // Show current state
        const state = lane.changeoverLeft > 0
            ? `<div style="margin-top:8px; font-size:11px; color:#f39c12;">Changeover: ${lane.changeoverLeft}s remaining</div>`
            : lane.cycleLeft > 0
            ? `<div style="margin-top:8px; font-size:11px; color:var(--accent);">Producing ${lane.producing}: ${lane.cycleLeft}s</div>`
            : lane.idx >= total
            ? `<div style="margin-top:8px; font-size:11px; color:var(--accent);">Complete</div>`
            : '';
        html += state;

        // Per-product completion summary
        html += '<div style="display:flex; gap:8px; margin-top:8px; flex-wrap:wrap;">';
        hjSimProducts.forEach(p => {
            const done = lane.completed[p.product] || 0;
            html += `<div style="font-size:10px; color:var(--text-dim);"><span style="display:inline-block; width:8px; height:8px; background:${p.color}; border-radius:2px; margin-right:3px;"></span>${p.product}: ${done}</div>`;
        });
        html += '</div>';

        el.innerHTML = html;
    }

    renderLane(hjSim.batched, 'hj-batched-visual');
    renderLane(hjSim.leveled, 'hj-leveled-visual');
}

function hjSimUpdateChart() {
    const h = hjSim.history;
    if (h.time.length < 2) return;

    Plotly.newPlot('hj-sim-chart', [
        { x: h.time, y: h.bWip, type: 'scatter', mode: 'lines', name: 'Batched WIP', line: { color: '#e74c3c', width: 2 } },
        { x: h.time, y: h.lWip, type: 'scatter', mode: 'lines', name: 'Leveled WIP', line: { color: '#4a9f6e', width: 2 } },
    ], {
        margin: { t: 20, b: 50, l: 50, r: 20 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a' },
        xaxis: { title: 'Time (s)', gridcolor: 'rgba(255,255,255,0.1)' },
        yaxis: { title: 'WIP (units)', gridcolor: 'rgba(255,255,255,0.1)' },
        legend: { orientation: 'h', y: -0.25, x: 0.5, xanchor: 'center' },
    }, { responsive: true, displayModeBar: false });
}

function hjSimUpdateInsights() {
    const b = hjSim.batched, l = hjSim.leveled;
    const bMaxWip = Math.max(...hjSim.history.bWip, 0);
    const lMaxWip = Math.max(...hjSim.history.lWip, 0);
    const wipReduction = bMaxWip > 0 ? ((1 - lMaxWip / bMaxWip) * 100).toFixed(0) : 0;
    const bAvgWip = hjSim.history.bWip.length > 0 ? (hjSim.history.bWip.reduce((a,b)=>a+b,0) / hjSim.history.bWip.length).toFixed(1) : 0;
    const lAvgWip = hjSim.history.lWip.length > 0 ? (hjSim.history.lWip.reduce((a,b)=>a+b,0) / hjSim.history.lWip.length).toFixed(1) : 0;

    let html = '<div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">';
    html += `<div><strong>Batched:</strong> Peak WIP = ${bMaxWip}, Avg WIP = ${bAvgWip}, Changeovers = ${b.changeovers}</div>`;
    html += `<div><strong>Leveled:</strong> Peak WIP = ${lMaxWip}, Avg WIP = ${lAvgWip}, Changeovers = ${l.changeovers}</div>`;
    html += '</div>';

    if (parseInt(wipReduction) > 0) {
        html += `<div style="margin-top:12px; padding:10px; background:rgba(74,159,110,0.15); border-radius:6px; font-weight:600; color:var(--accent);">
            Leveled production reduced peak WIP by ${wipReduction}% (${bMaxWip} → ${lMaxWip} units)
            ${l.changeovers > b.changeovers ? ` at the cost of ${l.changeovers - b.changeovers} additional changeovers.` : '.'}
        </div>`;
    } else if (parseInt(wipReduction) < 0) {
        html += `<div style="margin-top:12px; padding:10px; background:rgba(231,76,60,0.15); border-radius:6px; color:#e74c3c;">
            In this configuration, leveling didn't reduce WIP. The changeover time may be too long relative to cycle time — try reducing it with SMED.
        </div>`;
    }

    document.getElementById('hj-sim-insights').innerHTML = html;

    // Publish
    SvendOps.publish('hjSimBatchWip', bMaxWip, 'units', 'Heijunka Sim');
    SvendOps.publish('hjSimLevelWip', lMaxWip, 'units', 'Heijunka Sim');

    renderNextSteps('hj-sim-next-steps', [
        { title: 'SMED', desc: 'Reduce changeover time to enable more frequent leveling', calcId: 'smed' },
        { title: 'Mixed-Model', desc: 'Build the leveled production sequence', calcId: 'mixed-model' },
        { title: 'Line Simulator', desc: 'Simulate the production line with this product mix', calcId: 'line-sim' },
    ]);
    document.getElementById('hj-sim-next-steps').style.display = '';
}

// Init Heijunka Sim products on load
(function initHeijunkaSim() {
    const observer = new MutationObserver(() => {
        const el = document.getElementById('hj-sim-products');
        if (el && el.children.length === 0) hjSimRenderProducts();
    });
    document.addEventListener('DOMContentLoaded', () => {
        hjSimRenderProducts();
    });
})();
