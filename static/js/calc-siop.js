/**
 * calc-siop.js — SIOP (Sales, Inventory & Operations Planning) Calculators
 *
 * Load order: after calc-core.js
 * Extracted from: calculators.html (inline script)
 *
 * Contains: ABC Analysis, Demand Profile (Syntetos-Boylan),
 *           Service Level Trade-off, MRP Netting (Gross-to-Net)
 */

// =============================================================================
// ABC Analysis (Client-Side)
// =============================================================================
let abcItems = [];

function addABCItem(name, value) {
    abcItems.push({ name: name || `SKU-${String(abcItems.length + 1).padStart(3, '0')}`, value: value || 0 });
    renderABCItems();
}

function removeABCItem(idx) {
    abcItems.splice(idx, 1);
    renderABCItems();
}

function updateABCItem(idx, field, val) {
    if (field === 'value') val = parseFloat(val) || 0;
    abcItems[idx][field] = val;
    calcABC();
}

function loadSampleABCItems() {
    abcItems = [
        { name: 'SKU-001', value: 50000 }, { name: 'SKU-002', value: 42000 },
        { name: 'SKU-003', value: 38000 }, { name: 'SKU-004', value: 35000 },
        { name: 'SKU-005', value: 30000 }, { name: 'SKU-006', value: 12000 },
        { name: 'SKU-007', value: 11000 }, { name: 'SKU-008', value: 10000 },
        { name: 'SKU-009', value: 9000 },  { name: 'SKU-010', value: 8000 },
        { name: 'SKU-011', value: 3000 },  { name: 'SKU-012', value: 2800 },
        { name: 'SKU-013', value: 2500 },  { name: 'SKU-014', value: 2200 },
        { name: 'SKU-015', value: 2000 },
    ];
    renderABCItems();
}

function renderABCItems() {
    const container = document.getElementById('abc-items');
    if (!container) return;
    container.innerHTML = abcItems.map((item, i) => `
        <div class="yamazumi-station" style="gap: 8px;">
            <input type="text" value="${item.name}" style="flex: 1; padding: 8px;"
                   oninput="updateABCItem(${i}, 'name', this.value)" placeholder="SKU name">
            <input type="number" value="${item.value}" style="width: 100px; text-align: right;"
                   oninput="updateABCItem(${i}, 'value', this.value)" placeholder="Value">
            <button class="yamazumi-station-remove" onclick="removeABCItem(${i})">&times;</button>
        </div>
    `).join('');
    calcABC();
}

function calcABC() {
    const aPct = parseFloat(document.getElementById('abc-a-pct').value) || 80;
    const bPct = parseFloat(document.getElementById('abc-b-pct').value) || 95;

    const items = abcItems.filter(item => item.name && item.value > 0).map(i => ({...i}));
    if (items.length === 0) return;

    items.sort((a, b) => b.value - a.value);
    const totalValue = items.reduce((s, i) => s + i.value, 0);

    let cumulative = 0;
    items.forEach(item => {
        cumulative += item.value;
        item.cumPct = (cumulative / totalValue) * 100;
        item.category = item.cumPct <= aPct ? 'A' : item.cumPct <= bPct ? 'B' : 'C';
    });

    const aItems = items.filter(i => i.category === 'A');
    const bItems = items.filter(i => i.category === 'B');
    const cItems = items.filter(i => i.category === 'C');

    document.getElementById('abc-a-count').innerHTML = `${aItems.length}<span class="calc-result-unit"> (${(aItems.length/items.length*100).toFixed(0)}%)</span>`;
    document.getElementById('abc-b-count').innerHTML = `${bItems.length}<span class="calc-result-unit"> (${(bItems.length/items.length*100).toFixed(0)}%)</span>`;
    document.getElementById('abc-c-count').innerHTML = `${cItems.length}<span class="calc-result-unit"> (${(cItems.length/items.length*100).toFixed(0)}%)</span>`;

    const aVal = aItems.reduce((s,i) => s+i.value, 0);
    const bVal = bItems.reduce((s,i) => s+i.value, 0);
    const cVal = cItems.reduce((s,i) => s+i.value, 0);
    document.getElementById('abc-breakdown').innerHTML = `
        <div class="calc-breakdown-row"><span>A Items Value</span><span>$${Math.round(aVal).toLocaleString()} (${(aVal/totalValue*100).toFixed(1)}%)</span></div>
        <div class="calc-breakdown-row"><span>B Items Value</span><span>$${Math.round(bVal).toLocaleString()} (${(bVal/totalValue*100).toFixed(1)}%)</span></div>
        <div class="calc-breakdown-row"><span>C Items Value</span><span>$${Math.round(cVal).toLocaleString()} (${(cVal/totalValue*100).toFixed(1)}%)</span></div>
        <div class="calc-breakdown-row"><span>Total Value</span><span>$${Math.round(totalValue).toLocaleString()}</span></div>`;

    // Pareto chart
    const colors = items.map(i => i.category === 'A' ? '#4a9f6e' : i.category === 'B' ? '#e8c547' : '#e89547');
    ForgeViz.render(document.getElementById('abc-chart'), {
        title: '', chart_type: 'bar',
        traces: [
            { x: items.map(i => i.name), y: items.map(i => i.value), name: 'Value', trace_type: 'bar', color: colors },
            { x: items.map(i => i.name), y: items.map(i => i.cumPct), name: 'Cumulative %', trace_type: 'line', color: '#e74c3c', width: 2 }
        ],
        reference_lines: [
            { value: aPct, axis: 'y', color: '#4a9f6e', dash: 'dashed', label: 'A cutoff' },
            { value: bPct, axis: 'y', color: '#e8c547', dash: 'dashed', label: 'B cutoff' }
        ],
        x_axis: { label: '' }, y_axis: { label: 'Value ($)' }
    });

    SvendOps.publish('abcAItems', aItems.length, 'items', 'ABC Analysis');
    SvendOps.publish('abcBItems', bItems.length, 'items', 'ABC Analysis');
    SvendOps.publish('abcCItems', cItems.length, 'items', 'ABC Analysis');

    renderNextSteps('abc-next-steps', [
        { title: 'EOQ for A Items', calcId: 'eoq' },
        { title: 'Safety Stock', calcId: 'safety' },
        { title: 'Demand Profile', calcId: 'demand-profile' }
    ]);
}

// =============================================================================
// Demand Profile (Syntetos-Boylan Classification)
// =============================================================================
let dpPeriods = [];

function addDPPeriod(demand) {
    dpPeriods.push({ label: `P${dpPeriods.length + 1}`, demand: demand || 0 });
    renderDPPeriods();
}

function removeDPPeriod(idx) {
    dpPeriods.splice(idx, 1);
    dpPeriods.forEach((p, i) => p.label = `P${i + 1}`);
    renderDPPeriods();
}

function updateDPPeriod(idx, val) {
    dpPeriods[idx].demand = parseFloat(val) || 0;
    calcDemandProfile();
}

function loadSampleDPPeriods() {
    const sample = [120,0,130,0,125,140,0,135,150,0,145,160,0,155,170,0,165,180];
    dpPeriods = sample.map((d, i) => ({ label: `P${i + 1}`, demand: d }));
    renderDPPeriods();
}

function renderDPPeriods() {
    const container = document.getElementById('dp-periods');
    if (!container) return;
    container.innerHTML = dpPeriods.map((p, i) => `
        <div class="yamazumi-station" style="gap: 8px;">
            <span style="width:50px;color:var(--text-dim);font-size:12px;">${p.label}</span>
            <input type="number" value="${p.demand}" style="flex: 1; text-align: right; padding: 8px;"
                   oninput="updateDPPeriod(${i}, this.value)" placeholder="Demand">
            <button class="yamazumi-station-remove" onclick="removeDPPeriod(${i})">&times;</button>
        </div>
    `).join('');
    calcDemandProfile();
}

function calcDemandProfile() {
    const values = dpPeriods.map(p => p.demand);
    if (values.length < 3) return;

    // Non-zero demands only for CoV
    const nonZero = values.filter(v => v > 0);
    const mean = nonZero.reduce((s, v) => s + v, 0) / nonZero.length;
    const std = Math.sqrt(nonZero.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / nonZero.length);
    const cov = mean > 0 ? std / mean : 0;
    const covSq = cov * cov;

    // Average inter-demand interval (ADI)
    let intervals = [];
    let lastDemand = -1;
    for (let i = 0; i < values.length; i++) {
        if (values[i] > 0) {
            if (lastDemand >= 0) intervals.push(i - lastDemand);
            lastDemand = i;
        }
    }
    const adi = intervals.length > 0 ? intervals.reduce((s, v) => s + v, 0) / intervals.length : values.length;

    // Syntetos-Boylan classification
    let pattern;
    if (covSq < 0.49 && adi < 1.32) pattern = 'Smooth';
    else if (covSq >= 0.49 && adi < 1.32) pattern = 'Erratic';
    else if (covSq < 0.49 && adi >= 1.32) pattern = 'Intermittent';
    else pattern = 'Lumpy';

    const patternColors = { Smooth: '#4a9f6e', Erratic: '#e8c547', Intermittent: '#e89547', Lumpy: '#e74c3c' };
    document.getElementById('dp-pattern').innerHTML = `<span style="color:${patternColors[pattern]}">${pattern}</span>`;
    document.getElementById('dp-cov').textContent = cov.toFixed(3);
    document.getElementById('dp-adi').textContent = adi.toFixed(2);

    // Demand time series chart
    ForgeViz.render(document.getElementById('dp-chart'), {
        title: '', chart_type: 'bar',
        traces: [
            { x: values.map((_, i) => `P${i+1}`), y: values, name: 'Demand', trace_type: 'bar', color: values.map(v => v > 0 ? patternColors[pattern] : 'rgba(255,255,255,0.1)') },
            { x: values.map((_, i) => `P${i+1}`), y: Array(values.length).fill(mean), name: `Mean (${mean.toFixed(0)})`, trace_type: 'line', color: '#e74c3c', width: 1 }
        ],
        reference_lines: [],
        x_axis: { label: '' }, y_axis: { label: 'Demand' }
    });

    SvendOps.publish('demandPattern', pattern, '', 'Demand Profile');
    SvendOps.publish('demandMean', mean, 'units', 'Demand Profile');
    SvendOps.publish('demandCoV', cov, '', 'Demand Profile');

    renderNextSteps('dp-next-steps', [
        { title: 'Safety Stock', calcId: 'safety' },
        { title: 'ABC Analysis', calcId: 'abc' }
    ]);
}

// =============================================================================
// Service Level Trade-off
// =============================================================================
function calcServiceLevel() {
    const demandMean = parseFloat(document.getElementById('sl-demand').value) || 0;
    const demandStd = parseFloat(document.getElementById('sl-std').value) || 0;
    const lt = parseFloat(document.getElementById('sl-lt').value) || 1;
    const unitCost = parseFloat(document.getElementById('sl-unit').value) || 0;
    const holdPct = parseFloat(document.getElementById('sl-hold').value) || 0;
    const stockoutCost = parseFloat(document.getElementById('sl-stockout').value) || 0;

    const holdingCost = unitCost * (holdPct / 100);
    const ltDemandStd = demandStd * Math.sqrt(lt);

    // Standard normal inverse (approximation)
    function normInv(p) {
        if (p <= 0) return -4; if (p >= 1) return 4;
        const t = Math.sqrt(-2 * Math.log(p < 0.5 ? p : 1 - p));
        const c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
        const d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
        let z = t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t);
        return p < 0.5 ? -z : z;
    }

    // Sweep service levels
    const levels = [];
    const holdCosts = [];
    const stockCosts = [];
    const totalCosts = [];
    let minCost = Infinity, optLevel = 95, optSS = 0;

    for (let sl = 80; sl <= 99.9; sl += 0.5) {
        const z = normInv(sl / 100);
        const ss = z * ltDemandStd;
        const hc = ss * holdingCost;
        const expectedShortage = ltDemandStd * (Math.exp(-z*z/2) / Math.sqrt(2*Math.PI) - z * (1 - sl/100));
        const sc = Math.max(0, expectedShortage) * stockoutCost * (demandMean * 365 / (demandMean * lt || 1));
        const tc = hc + sc;

        levels.push(sl);
        holdCosts.push(hc);
        stockCosts.push(sc);
        totalCosts.push(tc);

        if (tc < minCost) { minCost = tc; optLevel = sl; optSS = ss; }
    }

    document.getElementById('sl-optimal').innerHTML = `${optLevel.toFixed(1)}<span class="calc-result-unit">%</span>`;
    document.getElementById('sl-ss').innerHTML = `${Math.round(optSS)}<span class="calc-result-unit">units</span>`;
    document.getElementById('sl-cost').textContent = `$${Math.round(minCost).toLocaleString()}/yr`;

    const optZ = normInv(optLevel / 100);
    document.getElementById('sl-breakdown').innerHTML = `
        <div class="calc-breakdown-row"><span>Z-score at optimal</span><span>${optZ.toFixed(2)}</span></div>
        <div class="calc-breakdown-row"><span>LT demand std</span><span>${ltDemandStd.toFixed(1)} units</span></div>
        <div class="calc-breakdown-row"><span>Holding cost at optimal</span><span>$${Math.round(optSS * holdingCost).toLocaleString()}/yr</span></div>
        <div class="calc-breakdown-row"><span>Formula</span><span>Min(Holding + Stockout cost) over SL range</span></div>`;

    ForgeViz.render(document.getElementById('sl-chart'), {
        title: '', chart_type: 'line',
        traces: [
            { x: levels, y: holdCosts, name: 'Holding Cost', trace_type: 'line', color: '#3a7f8f', width: 2 },
            { x: levels, y: stockCosts, name: 'Stockout Cost', trace_type: 'line', color: '#e89547', width: 2 },
            { x: levels, y: totalCosts, name: 'Total Cost', trace_type: 'line', color: '#4a9f6e', width: 3 }
        ],
        reference_lines: [
            { value: optLevel, axis: 'x', color: '#e74c3c', dash: 'dashed', label: `Optimal: ${optLevel.toFixed(1)}%` }
        ],
        markers: [{ x: optLevel, y: minCost, label: `Optimal: ${optLevel.toFixed(1)}%`, color: '#e74c3c' }],
        x_axis: { label: 'Service Level (%)' }, y_axis: { label: 'Annual Cost ($)' }
    });

    SvendOps.publish('optimalServiceLevel', optLevel, '%', 'Service Level');
    SvendOps.publish('optimalSafetyStock', Math.round(optSS), 'units', 'Service Level');

    renderNextSteps('sl-next-steps', [
        { title: 'Safety Stock Calculator', calcId: 'safety' },
        { title: 'EOQ', calcId: 'eoq' }
    ]);
}

// =============================================================================
// MRP Netting (Gross-to-Net)
// =============================================================================
let mrpPeriods = [];

function addMRPPeriod(gross, receipts) {
    mrpPeriods.push({ gross: gross || 0, receipts: receipts || 0 });
    renderMRPPeriods();
}

function removeMRPPeriod(idx) {
    mrpPeriods.splice(idx, 1);
    renderMRPPeriods();
}

function updateMRPPeriod(idx, field, val) {
    mrpPeriods[idx][field] = parseFloat(val) || 0;
    calcMRP();
}

function loadSampleMRPPeriods() {
    mrpPeriods = [
        { gross: 100, receipts: 0 }, { gross: 120, receipts: 100 },
        { gross: 80, receipts: 0 },  { gross: 150, receipts: 0 },
        { gross: 90, receipts: 0 },  { gross: 110, receipts: 0 },
    ];
    renderMRPPeriods();
}

function renderMRPPeriods() {
    const container = document.getElementById('mrp-periods');
    if (!container) return;
    container.innerHTML = mrpPeriods.map((p, i) => `
        <div class="yamazumi-station" style="gap: 8px;">
            <span style="width:50px;color:var(--text-dim);font-size:12px;">P${i + 1}</span>
            <input type="number" value="${p.gross}" style="flex: 1; text-align: right; padding: 8px;"
                   oninput="updateMRPPeriod(${i}, 'gross', this.value)" placeholder="Gross" title="Gross requirement">
            <input type="number" value="${p.receipts}" style="flex: 1; text-align: right; padding: 8px;"
                   oninput="updateMRPPeriod(${i}, 'receipts', this.value)" placeholder="Receipts" title="Scheduled receipts">
            <button class="yamazumi-station-remove" onclick="removeMRPPeriod(${i})">&times;</button>
        </div>
    `).join('');
    calcMRP();
}

function calcMRP() {
    const onHand = parseFloat(document.getElementById('mrp-oh').value) || 0;
    const ss = parseFloat(document.getElementById('mrp-ss').value) || 0;
    const lt = parseInt(document.getElementById('mrp-lt').value) || 1;
    const lotType = document.getElementById('mrp-lot').value;

    const gross = mrpPeriods.map(p => p.gross);
    const receipts = mrpPeriods.map(p => p.receipts);
    const n = gross.length;
    if (n === 0) return;

    // MRP netting
    const netReq = [];
    const plannedReceipt = [];
    const plannedRelease = Array(n).fill(0);
    const projected = [];
    let oh = onHand;
    let totalOrdered = 0;
    let orderCount = 0;

    for (let i = 0; i < n; i++) {
        const available = oh + receipts[i] + (plannedReceipt[i] || 0);
        const net = gross[i] + ss - available;
        netReq.push(Math.max(0, net));

        let orderQty = 0;
        if (net > 0) {
            orderQty = lotType === 'fixed' ? Math.ceil(net / 100) * 100 : net;
            orderCount++;
            totalOrdered += orderQty;
        }
        plannedReceipt[i] = (plannedReceipt[i] || 0) + orderQty;

        // Offset release by lead time
        const releaseIdx = i - lt;
        if (releaseIdx >= 0 && orderQty > 0) {
            plannedRelease[releaseIdx] = orderQty;
        }

        oh = available + orderQty - gross[i];
        projected.push(oh);
    }

    document.getElementById('mrp-orders').textContent = orderCount;
    document.getElementById('mrp-total').innerHTML = `${Math.round(totalOrdered).toLocaleString()}<span class="calc-result-unit">units</span>`;

    // MRP Grid table
    let gridHtml = `<table style="width:100%;border-collapse:collapse;font-size:12px;color:var(--text);">
        <tr style="border-bottom:1px solid var(--border);"><th style="text-align:left;padding:4px;">Period</th>`;
    for (let i = 0; i < n; i++) gridHtml += `<th style="padding:4px;text-align:right;">${i+1}</th>`;
    gridHtml += '</tr>';

    const rows = [
        ['Gross Req', gross],
        ['Sched Receipts', receipts],
        ['Net Req', netReq],
        ['Planned Receipt', plannedReceipt.slice(0, n)],
        ['Planned Release', plannedRelease],
        ['Projected OH', projected]
    ];
    rows.forEach(([label, data]) => {
        gridHtml += `<tr style="border-bottom:1px solid var(--border,rgba(255,255,255,0.1));"><td style="padding:4px;font-weight:500;">${label}</td>`;
        (data || []).forEach(v => {
            const val = Math.round(v || 0);
            const style = val < 0 ? 'color:#e74c3c;' : val > 0 && label === 'Planned Release' ? 'color:#4a9f6e;font-weight:600;' : '';
            gridHtml += `<td style="padding:4px;text-align:right;${style}">${val}</td>`;
        });
        gridHtml += '</tr>';
    });
    gridHtml += '</table>';
    document.getElementById('mrp-grid').innerHTML = gridHtml;

    // Chart
    ForgeViz.render(document.getElementById('mrp-chart'), {
        title: '', chart_type: 'bar',
        traces: [
            { x: gross.map((_, i) => `P${i+1}`), y: gross, name: 'Gross Req', trace_type: 'bar', color: '#e89547' },
            { x: gross.map((_, i) => `P${i+1}`), y: projected, name: 'Projected OH', trace_type: 'line', color: '#4a9f6e', width: 2 },
            { x: gross.map((_, i) => `P${i+1}`), y: Array(n).fill(ss), name: 'Safety Stock', trace_type: 'line', color: '#e74c3c', width: 1 }
        ],
        reference_lines: [
            { value: ss, axis: 'y', color: '#e74c3c', dash: 'dashed', label: 'Safety Stock' }
        ],
        x_axis: { label: '' }, y_axis: { label: 'Units' }
    });

    SvendOps.publish('mrpPlannedOrders', orderCount, 'orders', 'MRP Netting');

    renderNextSteps('mrp-next-steps', [
        { title: 'EOQ (lot sizing)', calcId: 'eoq' },
        { title: 'Kanban Sizing', calcId: 'kanban' }
    ]);
}

// Init new calculators on page load
document.addEventListener('DOMContentLoaded', () => {
    calcMTBF();
    calcErlang();
    riskRenderTable();
    riskUpdate();
    loadSampleABCItems();
    loadSampleDPPeriods();
    calcServiceLevel();
    loadSampleMRPPeriods();
});
