/**
 * calc-production.js — Production & Inventory Calculators
 *
 * Load order: after calc-core.js (uses SvendOps, renderNextSteps, MonteCarlo, showToast)
 * Extracted from: calculators.html (inline script)
 *
 * Provides: calcTakt, calcRTO, yamazumi, calcKanban, calcEPEI, calcSafety,
 * calcEOQ, calcOEE, bottleneck, calcLittles, calcPitch, calcTurns, calcCOQ,
 * calcLineEff, calcOLE, calcCycle, calcBA, calcHeijunka.
 */

// ============================================================================
// Takt Time
// ============================================================================

function calcTakt() {
    const available = parseFloat(document.getElementById('takt-available').value) || 0;
    const breaks = parseFloat(document.getElementById('takt-breaks').value) || 0;
    const demand = parseFloat(document.getElementById('takt-demand').value) || 1;

    const net = available - breaks;
    const taktMin = net / demand;
    const taktSec = taktMin * 60;

    document.getElementById('takt-result').innerHTML = `${taktMin.toFixed(2)}<span class="calc-result-unit">min</span>`;
    document.getElementById('takt-seconds').innerHTML = `${Math.round(taktSec)}<span class="calc-result-unit">sec</span>`;
    document.getElementById('takt-net').innerHTML = `${net}<span class="calc-result-unit">min</span>`;
    document.getElementById('takt-interpret').textContent = `${taktMin.toFixed(1)} min`;

    // Publish to shared state
    SvendOps.publish('takt', taktSec, 'sec', 'Takt Time');
    SvendOps.publish('taktMin', taktMin, 'min', 'Takt Time');

    // Update derivation
    document.getElementById('takt-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Calculate Net Available Time</div>
            <span class="formula">Net Time = Available − Breaks</span><br>
            Net Time = ${available} − ${breaks} = <strong>${net} min</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 2: Calculate Takt Time</div>
            <span class="formula">Takt = Net Time ÷ Demand</span><br>
            Takt = ${net} ÷ ${demand} = <strong>${taktMin.toFixed(2)} min</strong> = <strong>${Math.round(taktSec)} sec</strong>
        </div>
        <div class="step">
            <div class="step-num">Interpretation</div>
            To meet customer demand of <strong>${demand} units/shift</strong>,
            you must complete one unit every <strong>${taktMin.toFixed(1)} minutes</strong> (${Math.round(taktSec)} seconds).
        </div>
    `;

    // Takt gauge chart
    const gaugeMax = Math.max(taktMin * 2, 10);
    ForgeViz.render(document.getElementById('takt-chart'), {
        title: '', chart_type: 'gauge',
        gauge: { value: taktMin, min: 0, max: gaugeMax, label: 'min' },
        zones: [
            { low: 0, high: 1, axis: 'y', color: 'rgba(231,76,60,0.35)', label: '<1 min: high-speed risk' },
            { low: 1, high: 5, axis: 'y', color: 'rgba(74,159,110,0.35)', label: '1-5 min: optimal' },
            { low: 5, high: gaugeMax, axis: 'y', color: 'rgba(243,156,18,0.35)', label: '>5 min: batch risk' }
        ],
        traces: [], reference_lines: [], markers: [],
        x_axis: { label: '' }, y_axis: { label: '' }
    });

    renderNextSteps('takt-next-steps', [
        { title: 'Staff the Line', desc: 'Calculate operators needed at this takt', calcId: 'rto' },
        { title: 'Balance Stations', desc: 'Build a Yamazumi chart to balance work', calcId: 'yamazumi' },
        { title: 'Set Pitch', desc: 'Calculate pitch interval from takt', calcId: 'pitch' },
        { title: 'Simulate Flow', desc: 'Run a line simulation at this rate', calcId: 'linesim' },
    ]);
}

// ============================================================================
// RTO
// ============================================================================

function calcRTO() {
    const cycle = parseFloat(document.getElementById('rto-cycle').value) || 0;
    const takt = parseFloat(document.getElementById('rto-takt').value) || 1;
    const covPct = parseFloat(document.getElementById('rto-cov').value) || 0;

    const rto = cycle / takt;
    const cov = covPct / 100;
    const rtoMargin = rto * (1 + cov);
    const staff = Math.ceil(rtoMargin);
    const efficiency = (rto / staff) * 100;
    const buffer = rtoMargin - rto;

    document.getElementById('rto-base').textContent = rto.toFixed(2);
    document.getElementById('rto-margin').textContent = rtoMargin.toFixed(2);
    document.getElementById('rto-staff').innerHTML = `${staff}<span class="calc-result-unit">people</span>`;
    document.getElementById('rto-efficiency').textContent = `${efficiency.toFixed(1)}%`;
    document.getElementById('rto-buffer').textContent = `+${buffer.toFixed(2)} operators`;

    // Update derivation
    document.getElementById('rto-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Calculate Base RTO</div>
            <span class="formula">RTO = Cycle Time ÷ Takt Time</span><br>
            RTO = ${cycle} ÷ ${takt} = <strong>${rto.toFixed(2)} operators</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 2: Add Variation Margin</div>
            <span class="formula">RTO + Margin = RTO × (1 + CoV)</span><br>
            = ${rto.toFixed(2)} × (1 + ${cov.toFixed(2)}) = <strong>${rtoMargin.toFixed(2)} operators</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 3: Round Up to Whole People</div>
            <span class="formula">Staff = ⌈RTO + Margin⌉</span><br>
            Staff = ⌈${rtoMargin.toFixed(2)}⌉ = <strong>${staff} people</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 4: Calculate Line Efficiency</div>
            <span class="formula">Efficiency = (Base RTO ÷ Staff) × 100</span><br>
            = (${rto.toFixed(2)} ÷ ${staff}) × 100 = <strong>${efficiency.toFixed(1)}%</strong>
        </div>
    `;

    // Publish to shared state
    SvendOps.publish('rtoStaff', staff, 'people', 'RTO');
    SvendOps.publish('lineEfficiency', efficiency, '%', 'RTO');
}

// ============================================================================
// Yamazumi
// ============================================================================

let yamazumiData = [
    { name: 'Station 1', time: 45 },
    { name: 'Station 2', time: 50 },
    { name: 'Station 3', time: 55 },
];

function renderYamazumiInputs() {
    const container = document.getElementById('yamazumi-stations');
    container.innerHTML = yamazumiData.map((s, i) => `
        <div class="yamazumi-station">
            <span class="yamazumi-station-name">${s.name}</span>
            <input type="number" value="${s.time}" oninput="updateYamazumi(${i}, this.value)">
            <span style="color: var(--text-dim);">sec</span>
            <button class="yamazumi-station-remove" onclick="removeYamazumiStation(${i})">&times;</button>
        </div>
    `).join('');
    renderYamazumi();
}

function addYamazumiStation() {
    yamazumiData.push({ name: `Station ${yamazumiData.length + 1}`, time: 30 });
    renderYamazumiInputs();
}

function removeYamazumiStation(idx) {
    yamazumiData.splice(idx, 1);
    renderYamazumiInputs();
}

function updateYamazumi(idx, value) {
    yamazumiData[idx].time = parseFloat(value) || 0;
    renderYamazumi();
}

function renderYamazumi() {
    const takt = parseFloat(document.getElementById('yama-takt').value) || 60;
    const times = yamazumiData.map(s => s.time);
    const names = yamazumiData.map(s => s.name);

    // Calculate free capacity (time below takt)
    const freeCapacity = times.map(t => Math.max(0, takt - t));
    const utilization = times.map(t => Math.min(100, (t / takt) * 100));

    // Work time bar (bottom of stack)
    const workTrace = {
        x: names,
        y: times,
        type: 'bar',
        name: 'Work',
        marker: { color: times.map(t => t > takt ? '#e74c3c' : '#4a9f6e') },
        text: times.map((t, i) => `${t}s (${utilization[i].toFixed(0)}%)`),
        textposition: 'inside',
        textfont: { color: '#fff', size: 11 },
        hovertemplate: '%{x}<br>Work: %{y}s<br>Utilization: %{text}<extra></extra>',
    };

    // Free capacity bar (top of stack) - only for stations under takt
    const capacityTrace = {
        x: names,
        y: freeCapacity,
        type: 'bar',
        name: 'Free Capacity',
        marker: { color: 'rgba(255,255,255,0.15)' },
        text: freeCapacity.map(f => f > 0 ? `+${f.toFixed(0)}s` : ''),
        textposition: 'inside',
        textfont: { color: 'rgba(255,255,255,0.6)', size: 10 },
        hovertemplate: '%{x}<br>Free: %{y}s<extra></extra>',
    };

    ForgeViz.render(document.getElementById('yamazumi-chart'), {
        title: '', chart_type: 'stacked_bar',
        traces: [
            { x: names, y: times, name: 'Work Content', trace_type: 'bar', color: workTrace.marker.color },
            { x: names, y: freeCapacity, name: 'Free Capacity', trace_type: 'bar', color: 'rgba(255,255,255,0.15)' }
        ],
        reference_lines: [
            { value: takt, axis: 'y', color: '#e74c3c', dash: 'dashed', label: `Takt: ${takt}s` }
        ],
        zones: [], markers: [],
        y_axis: { label: 'Cycle Time (sec)' }, x_axis: { label: '' }
    });

    const total = times.reduce((a, b) => a + b, 0);
    const totalFree = freeCapacity.reduce((a, b) => a + b, 0);
    const rto = total / takt;
    const efficiency = (rto / yamazumiData.length) * 100;
    const avgUtil = utilization.reduce((a, b) => a + b, 0) / utilization.length;

    document.getElementById('yama-total').innerHTML = `${total}<span class="calc-result-unit">sec</span>`;
    document.getElementById('yama-rto').textContent = rto.toFixed(2);
    document.getElementById('yama-efficiency').innerHTML = `${efficiency.toFixed(1)}<span class="calc-result-unit">%</span>`;
    document.getElementById('yama-free').innerHTML = `${totalFree.toFixed(0)}<span class="calc-result-unit">sec</span>`;
    document.getElementById('yama-util').innerHTML = `${avgUtil.toFixed(1)}<span class="calc-result-unit">%</span>`;
}

// ============================================================================
// Kanban
// ============================================================================

function calcKanban() {
    const demand = parseFloat(document.getElementById('kanban-demand').value) || 0;
    const lead = parseFloat(document.getElementById('kanban-lead').value) || 0;
    const safetyPct = parseFloat(document.getElementById('kanban-safety').value) || 0;
    const container = parseFloat(document.getElementById('kanban-container').value) || 1;

    const base = demand * lead;
    const safety = base * (safetyPct / 100);
    const total = base + safety;
    const cards = Math.ceil(total / container);

    document.getElementById('kanban-base').innerHTML = `${Math.round(base)}<span class="calc-result-unit">units</span>`;
    document.getElementById('kanban-ss').innerHTML = `${Math.round(safety)}<span class="calc-result-unit">units</span>`;
    document.getElementById('kanban-cards').innerHTML = `${cards}<span class="calc-result-unit">cards</span>`;
    document.getElementById('kanban-total').textContent = `${Math.round(total)} units`;

    // Update derivation
    document.getElementById('kanban-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Calculate Base Stock</div>
            <span class="formula">Base = Demand × Lead Time</span><br>
            Base = ${demand} × ${lead} = <strong>${Math.round(base)} units</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 2: Calculate Safety Stock</div>
            <span class="formula">Safety = Base × Safety Factor %</span><br>
            Safety = ${Math.round(base)} × ${safetyPct}% = <strong>${Math.round(safety)} units</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 3: Calculate Kanban Cards</div>
            <span class="formula">Cards = ⌈(Base + Safety) ÷ Container Size⌉</span><br>
            Cards = ⌈(${Math.round(base)} + ${Math.round(safety)}) ÷ ${container}⌉ = ⌈${(total / container).toFixed(2)}⌉ = <strong>${cards} cards</strong>
        </div>
        <div class="step">
            <div class="step-num">Interpretation</div>
            Total inventory in the loop: <strong>${Math.round(total)} units</strong> across ${cards} containers of ${container} units each.
        </div>
    `;

    // Publish to shared state
    SvendOps.publish('kanbanCards', cards, 'cards', 'Kanban');
    SvendOps.publish('kanbanInventory', Math.round(total), 'units', 'Kanban');

    // Kanban pipeline visual
    const baseCards = Math.ceil(base / container);
    const safetyCards = cards - baseCards;
    const baseTokens = Array(Math.min(baseCards, 12)).fill('<span style="display:inline-block;width:18px;height:24px;background:#4a9f6e;border-radius:3px;margin:2px;"></span>').join('');
    const safetyTokens = Array(Math.min(safetyCards, 6)).fill('<span style="display:inline-block;width:18px;height:24px;background:#f39c12;border-radius:3px;margin:2px;"></span>').join('');
    const vis = document.getElementById('kanban-visual');
    vis.innerHTML = `
        <div style="display:flex;align-items:center;gap:16px;justify-content:center;flex-wrap:wrap;">
            <div style="text-align:center;padding:12px 16px;border:1px solid var(--border);border-radius:8px;background:var(--bg-secondary);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#9aaa9a" stroke-width="2"><rect x="1" y="3" width="15" height="13"/><polygon points="16 8 20 8 23 11 23 16 16 16 16 8"/><circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/></svg>
                <div style="font-size:11px;color:var(--text-dim);margin-top:4px;">Supplier</div>
            </div>
            <div style="font-size:20px;color:var(--text-dim);">→</div>
            <div style="text-align:center;padding:8px 12px;min-width:80px;">
                <div style="display:flex;flex-wrap:wrap;justify-content:center;max-width:240px;">${baseTokens}${safetyTokens}</div>
                <div style="font-size:11px;color:var(--text-dim);margin-top:6px;">
                    <span style="color:#4a9f6e;">${baseCards} base</span> + <span style="color:#f39c12;">${safetyCards} safety</span> = ${cards} cards
                </div>
                <div style="font-size:10px;color:var(--text-dim);">${cards} × ${container} = ${Math.round(total)} units in loop</div>
            </div>
            <div style="font-size:20px;color:var(--text-dim);">→</div>
            <div style="text-align:center;padding:12px 16px;border:1px solid var(--border);border-radius:8px;background:var(--bg-secondary);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#9aaa9a" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
                <div style="font-size:11px;color:var(--text-dim);margin-top:4px;">Customer</div>
            </div>
        </div>`;
    vis.style.display = 'block';
}

function toggleKanbanMonteCarlo(btn) {
    btn.classList.toggle('active');
    const results = document.getElementById('kanban-monte-results');
    results.classList.toggle('visible');
    if (btn.classList.contains('active')) { runKanbanMonteCarlo(); }
}

function runKanbanMonteCarlo() {
    const demand = parseFloat(document.getElementById('kanban-demand').value) || 0;
    const lead = parseFloat(document.getElementById('kanban-lead').value) || 0;
    const safetyPct = parseFloat(document.getElementById('kanban-safety').value) || 0;
    const container = parseFloat(document.getElementById('kanban-container').value) || 1;

    const calcCards = (d, l, sf) => {
        const base = d * l;
        const safety = base * (sf / 100);
        return Math.ceil((base + safety) / container);
    };

    const inputs = [
        { value: demand, cv: 0.1 },
        { value: lead, cv: 0.15 },
        { value: safetyPct, cv: 0.1 }
    ];

    const sim = MonteCarlo.simulate(calcCards, inputs, 2000);

    document.getElementById('kanban-monte-mean').textContent = sim.mean.toFixed(1);
    document.getElementById('kanban-monte-p5').textContent = Math.round(sim.p5);
    document.getElementById('kanban-monte-p95').textContent = Math.round(sim.p95);
    document.getElementById('kanban-monte-std').textContent = `±${sim.std.toFixed(1)}`;

    MonteCarlo.renderHistogram('kanban-monte-chart', sim, 'Kanban Cards Distribution (2000 runs)', 'cards');
}

// ============================================================================
// EPEI
// ============================================================================

function calcEPEI() {
    const available = parseFloat(document.getElementById('epei-available').value) || 0;
    const parts = parseFloat(document.getElementById('epei-parts').value) || 1;
    const changeover = parseFloat(document.getElementById('epei-changeover').value) || 0;
    const targetPct = parseFloat(document.getElementById('epei-target').value) || 10;

    const totalCO = parts * changeover;
    const budget = available * 60 * (targetPct / 100);
    const epei = totalCO / budget;

    document.getElementById('epei-result').innerHTML = `${epei.toFixed(1)}<span class="calc-result-unit">days</span>`;
    document.getElementById('epei-total').innerHTML = `${totalCO}<span class="calc-result-unit">min</span>`;
    document.getElementById('epei-budget').innerHTML = `${Math.round(budget)}<span class="calc-result-unit">min</span>`;
    document.getElementById('epei-interpret').textContent = epei.toFixed(1);

    // Update derivation
    document.getElementById('epei-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Total Changeover Time per Cycle</div>
            <span class="formula">Total C/O = Parts × Changeover Time</span><br>
            Total C/O = ${parts} × ${changeover} = <strong>${totalCO} min</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 2: Changeover Budget per Day</div>
            <span class="formula">Budget = Available Hours × 60 × Target %</span><br>
            Budget = ${available} × 60 × ${targetPct}% = <strong>${Math.round(budget)} min/day</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 3: Calculate EPEI</div>
            <span class="formula">EPEI = Total C/O ÷ Daily Budget</span><br>
            EPEI = ${totalCO} ÷ ${Math.round(budget)} = <strong>${epei.toFixed(1)} days</strong>
        </div>
        <div class="step">
            <div class="step-num">Interpretation</div>
            With ${parts} parts and ${changeover} min per changeover, you can run <strong>Every Part Every ${epei.toFixed(1)} days</strong> within the ${targetPct}% time budget.
        </div>
    `;

    // Publish to shared state
    SvendOps.publish('epei', epei, 'days', 'EPEI');

    renderNextSteps('epei-next-steps', [
        { title: 'Build Heijunka Box', desc: 'Create a level schedule from EPEI', calcId: 'heijunka' },
        { title: 'Run SMED Event', desc: 'Reduce changeover to shorten EPEI', calcId: 'smed' },
    ]);
}

// ============================================================================
// Safety Stock
// ============================================================================

function calcSafety() {
    const demand = parseFloat(document.getElementById('safety-demand').value) || 0;
    const demandStd = parseFloat(document.getElementById('safety-demand-std').value) || 0;
    const lead = parseFloat(document.getElementById('safety-lead').value) || 0;
    const leadStd = parseFloat(document.getElementById('safety-lead-std').value) || 0;
    const z = parseFloat(document.getElementById('safety-service').value) || 1.65;

    const combined = Math.sqrt((lead * demandStd * demandStd) + (demand * demand * leadStd * leadStd));
    const safety = z * combined;
    const rop = (demand * lead) + safety;
    const meanDemand = demand * lead;

    document.getElementById('safety-result').innerHTML = `${Math.round(safety)}<span class="calc-result-unit">units</span>`;
    document.getElementById('safety-rop').innerHTML = `${Math.round(rop)}<span class="calc-result-unit">units</span>`;
    document.getElementById('safety-z').textContent = z.toFixed(2);
    document.getElementById('safety-combined').textContent = `${Math.round(combined)} units`;

    // Normal distribution curve
    const xMin = meanDemand - 4 * combined;
    const xMax = meanDemand + 4 * combined;
    const step = (xMax - xMin) / 100;
    const xVals = [];
    const yVals = [];
    const fillX = [];
    const fillY = [];

    for (let x = xMin; x <= xMax; x += step) {
        const y = Math.exp(-0.5 * Math.pow((x - meanDemand) / combined, 2)) / (combined * Math.sqrt(2 * Math.PI));
        xVals.push(x);
        yVals.push(y);
        if (x <= rop) {
            fillX.push(x);
            fillY.push(y);
        }
    }

    ForgeViz.render(document.getElementById('safety-chart'), {
        title: '', chart_type: 'line',
        traces: [
            { x: fillX, y: fillY, name: 'Service Level', trace_type: 'area', color: 'rgba(74,159,110,0.4)', fill: 'tozeroy' },
            { x: xVals, y: yVals, name: 'Demand Distribution', trace_type: 'line', color: '#4a9f6e', width: 2 }
        ],
        reference_lines: [
            { value: meanDemand, axis: 'x', color: '#3a7f8f', dash: 'dashed', label: `Mean: ${Math.round(meanDemand)}` },
            { value: rop, axis: 'x', color: '#e74c3c', label: `ROP: ${Math.round(rop)}` }
        ],
        zones: [], markers: [],
        x_axis: { label: 'Demand During Lead Time (units)' },
        y_axis: { label: 'Probability' }
    });

    // Update derivation
    document.getElementById('safety-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Mean Demand During Lead Time</div>
            <span class="formula">Mean = Demand × Lead Time</span><br>
            Mean = ${demand} × ${lead} = <strong>${Math.round(meanDemand)} units</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 2: Combined Standard Deviation</div>
            <span class="formula">σ_combined = √(LT × σ_d² + D² × σ_LT²)</span><br>
            = √(${lead} × ${demandStd}² + ${demand}² × ${leadStd}²)<br>
            = √(${(lead * demandStd * demandStd).toFixed(0)} + ${(demand * demand * leadStd * leadStd).toFixed(0)}) = <strong>${combined.toFixed(1)} units</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 3: Safety Stock</div>
            <span class="formula">Safety Stock = Z × σ_combined</span><br>
            = ${z.toFixed(2)} × ${combined.toFixed(1)} = <strong>${Math.round(safety)} units</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 4: Reorder Point</div>
            <span class="formula">ROP = Mean Demand + Safety Stock</span><br>
            ROP = ${Math.round(meanDemand)} + ${Math.round(safety)} = <strong>${Math.round(rop)} units</strong>
        </div>
    `;

    // Publish to shared state
    SvendOps.publish('safetyStock', Math.round(safety), 'units', 'Safety Stock');
    SvendOps.publish('rop', Math.round(rop), 'units', 'Safety Stock');

    renderNextSteps('safety-next-steps', [
        { title: 'Size Kanbans', desc: 'Calculate kanban cards including safety stock', calcId: 'kanban' },
        { title: 'Set Reorder Qty', desc: 'Find optimal order quantity with EOQ', calcId: 'eoq' },
        { title: 'Check Turns', desc: 'Evaluate inventory turns performance', calcId: 'turns' },
    ]);
}

function toggleSafetyMonteCarlo(btn) {
    btn.classList.toggle('active');
    const results = document.getElementById('safety-monte-results');
    results.classList.toggle('visible');
    if (btn.classList.contains('active')) { runSafetyMonteCarlo(); }
}

function runSafetyMonteCarlo() {
    const demand = parseFloat(document.getElementById('safety-demand').value) || 0;
    const demandStd = parseFloat(document.getElementById('safety-demand-std').value) || 0;
    const lead = parseFloat(document.getElementById('safety-lead').value) || 0;
    const leadStd = parseFloat(document.getElementById('safety-lead-std').value) || 0;
    const z = parseFloat(document.getElementById('safety-service').value) || 1.65;

    const calcSS = (d, ds, l, ls) => {
        return z * Math.sqrt(l * ds * ds + d * d * ls * ls);
    };

    const inputs = [
        { value: demand, cv: 0.1 },
        { value: demandStd, cv: 0.15 },
        { value: lead, cv: 0.1 },
        { value: leadStd, cv: 0.15 }
    ];

    const sim = MonteCarlo.simulate(calcSS, inputs, 2000);

    document.getElementById('safety-monte-mean').textContent = Math.round(sim.mean);
    document.getElementById('safety-monte-p5').textContent = Math.round(sim.p5);
    document.getElementById('safety-monte-p95').textContent = Math.round(sim.p95);
    document.getElementById('safety-monte-std').textContent = `±${Math.round(sim.std)}`;

    MonteCarlo.renderHistogram('safety-monte-chart', sim, 'Safety Stock Distribution (2000 runs)', 'units');
}

// ============================================================================
// EOQ
// ============================================================================

function calcEOQ() {
    const demand = parseFloat(document.getElementById('eoq-demand').value) || 0;
    const orderCost = parseFloat(document.getElementById('eoq-order').value) || 0;
    const unitCost = parseFloat(document.getElementById('eoq-unit').value) || 0;
    const holdingPct = parseFloat(document.getElementById('eoq-holding').value) || 0;

    const holding = unitCost * (holdingPct / 100);
    const eoq = Math.sqrt((2 * demand * orderCost) / holding);
    const orders = demand / eoq;
    const annualOrder = orders * orderCost;
    const annualHold = (eoq / 2) * holding;
    const total = annualOrder + annualHold;

    document.getElementById('eoq-result').innerHTML = `${Math.round(eoq)}<span class="calc-result-unit">units</span>`;
    document.getElementById('eoq-orders').textContent = Math.round(orders);
    document.getElementById('eoq-order-cost').textContent = `$${Math.round(annualOrder).toLocaleString()}`;
    document.getElementById('eoq-hold-cost').textContent = `$${Math.round(annualHold).toLocaleString()}`;
    document.getElementById('eoq-total').textContent = `$${Math.round(total).toLocaleString()}`;

    // EOQ Cost Curve
    const minQ = Math.max(50, eoq * 0.2);
    const maxQ = eoq * 3;
    const step = (maxQ - minQ) / 50;
    const quantities = [];
    const orderCosts = [];
    const holdCosts = [];
    const totalCosts = [];

    for (let q = minQ; q <= maxQ; q += step) {
        quantities.push(Math.round(q));
        const oc = (demand / q) * orderCost;
        const hc = (q / 2) * holding;
        orderCosts.push(oc);
        holdCosts.push(hc);
        totalCosts.push(oc + hc);
    }

    ForgeViz.render(document.getElementById('eoq-chart'), {
        title: '', chart_type: 'line',
        traces: [
            { x: quantities, y: orderCosts, name: 'Order Cost', trace_type: 'line', color: '#3a7f8f', width: 2 },
            { x: quantities, y: holdCosts, name: 'Holding Cost', trace_type: 'line', color: '#e89547', width: 2 },
            { x: quantities, y: totalCosts, name: 'Total Cost', trace_type: 'line', color: '#4a9f6e', width: 3 }
        ],
        reference_lines: [
            { value: eoq, axis: 'x', color: '#e74c3c', dash: 'dashed', label: `EOQ: ${Math.round(eoq)}` }
        ],
        markers: [{ x: eoq, y: total, label: `EOQ: ${Math.round(eoq)}`, color: '#e74c3c' }],
        zones: [],
        x_axis: { label: 'Order Quantity' },
        y_axis: { label: 'Annual Cost ($)' }
    });

    // Update derivation
    document.getElementById('eoq-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Calculate Annual Holding Cost Rate</div>
            <span class="formula">H = Unit Cost × Holding %</span><br>
            H = $${unitCost} × ${holdingPct}% = <strong>$${holding.toFixed(2)}/unit/year</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 2: Apply EOQ Formula</div>
            <span class="formula">EOQ = √(2DS / H)</span><br>
            Where D = ${demand.toLocaleString()} units/year, S = $${orderCost}/order, H = $${holding.toFixed(2)}/unit/year<br>
            EOQ = √(2 × ${demand.toLocaleString()} × ${orderCost} / ${holding.toFixed(2)})<br>
            EOQ = √(${(2 * demand * orderCost).toLocaleString()} / ${holding.toFixed(2)})<br>
            EOQ = √${Math.round(2 * demand * orderCost / holding).toLocaleString()} = <strong>${Math.round(eoq)} units</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 3: Verify Cost Balance</div>
            <span class="formula">At EOQ, Order Cost = Holding Cost (optimal point)</span><br>
            Order Cost = (D/Q) × S = (${demand.toLocaleString()}/${Math.round(eoq)}) × $${orderCost} = <strong>$${Math.round(annualOrder).toLocaleString()}</strong><br>
            Holding Cost = (Q/2) × H = (${Math.round(eoq)}/2) × $${holding.toFixed(2)} = <strong>$${Math.round(annualHold).toLocaleString()}</strong><br>
            ✓ Costs are balanced at the optimal point
        </div>
    `;

    // Publish to shared state
    SvendOps.publish('eoq', Math.round(eoq), 'units', 'EOQ');
}

function toggleEOQMonteCarlo(btn) {
    btn.classList.toggle('active');
    const results = document.getElementById('eoq-monte-results');
    results.classList.toggle('visible');

    if (btn.classList.contains('active')) {
        runEOQMonteCarlo();
    }
}

function runEOQMonteCarlo() {
    const demand = parseFloat(document.getElementById('eoq-demand').value) || 0;
    const orderCost = parseFloat(document.getElementById('eoq-order').value) || 0;
    const unitCost = parseFloat(document.getElementById('eoq-unit').value) || 0;
    const holdingPct = parseFloat(document.getElementById('eoq-holding').value) || 0;

    // EOQ formula with variable inputs
    const calcEOQValue = (d, s, h) => {
        const holding = unitCost * (h / 100);
        return Math.sqrt((2 * d * s) / holding);
    };

    const inputs = [
        { value: demand, cv: 0.1 },      // ±10% demand variability
        { value: orderCost, cv: 0.1 },   // ±10% order cost variability
        { value: holdingPct, cv: 0.1 }   // ±10% holding cost variability
    ];

    const sim = MonteCarlo.simulate(calcEOQValue, inputs, 2000);

    document.getElementById('eoq-monte-mean').textContent = Math.round(sim.mean);
    document.getElementById('eoq-monte-p5').textContent = Math.round(sim.p5);
    document.getElementById('eoq-monte-p95').textContent = Math.round(sim.p95);
    document.getElementById('eoq-monte-std').textContent = `±${Math.round(sim.std)}`;

    MonteCarlo.renderHistogram('eoq-monte-chart', sim, 'EOQ Distribution (2000 runs)', 'units');
}

// ============================================================================
// OEE
// ============================================================================

function calcOEE() {
    const planned = parseFloat(document.getElementById('oee-planned').value) || 0;
    const downtime = parseFloat(document.getElementById('oee-downtime').value) || 0;
    const ideal = parseFloat(document.getElementById('oee-ideal').value) || 1;
    const produced = parseFloat(document.getElementById('oee-produced').value) || 0;
    const defects = parseFloat(document.getElementById('oee-defects').value) || 0;

    const runtime = planned - downtime;
    const good = produced - defects;
    const availability = runtime / planned;
    const performance = Math.min(1, (produced * ideal / 60) / runtime);
    const quality = good / produced;
    const oee = availability * performance * quality * 100;

    const oeeEl = document.getElementById('oee-result');
    oeeEl.textContent = `${oee.toFixed(1)}%`;
    oeeEl.className = 'oee-value ' + (oee >= 85 ? 'oee-world-class' : oee >= 65 ? 'oee-good' : 'oee-poor');

    document.getElementById('oee-availability').textContent = `${(availability * 100).toFixed(1)}%`;
    document.getElementById('oee-performance').textContent = `${(performance * 100).toFixed(1)}%`;
    document.getElementById('oee-quality').textContent = `${(quality * 100).toFixed(1)}%`;
    document.getElementById('oee-runtime').textContent = `${runtime} min`;
    document.getElementById('oee-good').textContent = `${good} units`;

    // Update derivation
    document.getElementById('oee-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Availability</div>
            <span class="formula">Availability = (Planned − Downtime) ÷ Planned</span><br>
            = (${planned} − ${downtime}) ÷ ${planned} = ${runtime} ÷ ${planned} = <strong>${(availability * 100).toFixed(1)}%</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 2: Performance</div>
            <span class="formula">Performance = (Produced × Ideal Cycle) ÷ Runtime</span><br>
            = (${produced} × ${ideal}/60) ÷ ${runtime} = <strong>${(performance * 100).toFixed(1)}%</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 3: Quality</div>
            <span class="formula">Quality = (Produced − Defects) ÷ Produced</span><br>
            = (${produced} − ${defects}) ÷ ${produced} = ${good} ÷ ${produced} = <strong>${(quality * 100).toFixed(1)}%</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 4: OEE</div>
            <span class="formula">OEE = Availability × Performance × Quality</span><br>
            = ${(availability * 100).toFixed(1)}% × ${(performance * 100).toFixed(1)}% × ${(quality * 100).toFixed(1)}% = <strong>${oee.toFixed(1)}%</strong>
        </div>
    `;

    // OEE Loss Breakdown Donut Chart
    const availLoss = (1 - availability) * 100;
    const perfLoss = availability * (1 - performance) * 100;
    const qualLoss = availability * performance * (1 - quality) * 100;

    const oeeColor = oee >= 85 ? '#27ae60' : oee >= 65 ? '#f39c12' : '#e74c3c';
    ForgeViz.render(document.getElementById('oee-chart'), {
        title: '', chart_type: 'donut',
        traces: [{ type: 'donut',
            labels: ['OEE', 'Avail Loss', 'Perf Loss', 'Quality Loss'],
            values: [oee, availLoss, perfLoss, qualLoss],
            colors: [oeeColor, '#e74c3c', '#f39c12', '#9b59b6'],
            center_label: `${oee.toFixed(0)}%`
        }],
        reference_lines: [], zones: [], markers: [],
        x_axis: { label: '' }, y_axis: { label: '' }
    });

    // Publish to shared state
    SvendOps.publish('oee', oee, '%', 'OEE');
    SvendOps.publish('oeeAvailability', parseFloat((availability * 100).toFixed(1)), '%', 'OEE');

    renderNextSteps('oee-next-steps', [
        { title: 'Bottleneck Analysis', desc: 'Find the constraint driving downtime', calcId: 'bottleneck' },
        { title: 'SMED', desc: 'Reduce changeover to improve availability', calcId: 'smed' },
        { title: 'Line Efficiency', desc: 'Check labor utilization on this line', calcId: 'lineeff' },
    ]);
}

// ============================================================================
// Bottleneck
// ============================================================================

let bottleneckData = [
    { name: 'Cutting', time: 45 },
    { name: 'Welding', time: 72 },
    { name: 'Assembly', time: 58 },
    { name: 'Testing', time: 40 },
];

function renderBottleneckInputs() {
    const container = document.getElementById('bottleneck-stations');
    container.innerHTML = bottleneckData.map((s, i) => `
        <div class="bottleneck-station">
            <input type="text" value="${s.name}" oninput="updateBottleneckName(${i}, this.value)">
            <input type="number" value="${s.time}" oninput="updateBottleneckTime(${i}, this.value)">
            <span style="color: var(--text-dim);">sec</span>
            <button class="yamazumi-station-remove" onclick="removeBottleneckStation(${i})">&times;</button>
        </div>
    `).join('');
    renderBottleneck();
}

function addBottleneckStation() {
    bottleneckData.push({ name: `Step ${bottleneckData.length + 1}`, time: 30 });
    renderBottleneckInputs();
}

function removeBottleneckStation(idx) {
    bottleneckData.splice(idx, 1);
    renderBottleneckInputs();
}

function updateBottleneckName(idx, value) {
    bottleneckData[idx].name = value;
    renderBottleneck();
}

function updateBottleneckTime(idx, value) {
    bottleneckData[idx].time = parseFloat(value) || 0;
    renderBottleneck();
}

function renderBottleneck() {
    if (bottleneckData.length === 0) return;

    const maxTime = Math.max(...bottleneckData.map(s => s.time));
    const bottleneckIdx = bottleneckData.findIndex(s => s.time === maxTime);
    const colors = bottleneckData.map((s, i) => i === bottleneckIdx ? '#e74c3c' : '#4a9f6e');

    ForgeViz.render(document.getElementById('bottleneck-chart'), {
        title: '', chart_type: 'bar',
        traces: [{ x: bottleneckData.map(s => s.name), y: bottleneckData.map(s => s.time),
            name: '', trace_type: 'bar', color: colors }],
        reference_lines: [], zones: [], markers: [],
        y_axis: { label: 'Cycle Time (sec)' }, x_axis: { label: '' }
    });

    const constraint = bottleneckData[bottleneckIdx];
    document.getElementById('bottleneck-name').textContent = constraint.name;
    document.getElementById('bottleneck-ct').textContent = `${constraint.time} sec`;
    document.getElementById('bottleneck-throughput').textContent = `${(3600 / constraint.time).toFixed(1)} units/hr`;

    // Publish to shared state
    SvendOps.publish('bottleneckCT', constraint.time, 'sec', 'Bottleneck');
    SvendOps.publish('bottleneckThroughput', parseFloat((3600 / constraint.time).toFixed(1)), 'units/hr', 'Bottleneck');
}

// ============================================================================
// Little's Law
// ============================================================================

function calcLittles() {
    const solve = document.getElementById('littles-solve').value;
    const wip = parseFloat(document.getElementById('littles-wip').value) || 0;
    const thr = parseFloat(document.getElementById('littles-thr').value) || 1;
    const lt = parseFloat(document.getElementById('littles-lt').value) || 1;

    // Show/hide input groups based on solve mode
    document.getElementById('littles-wip-group').style.display = solve === 'wip' ? 'none' : 'flex';
    document.getElementById('littles-thr-group').style.display = solve === 'throughput' ? 'none' : 'flex';
    document.getElementById('littles-lt-group').style.display = solve === 'leadtime' ? 'none' : 'flex';

    let result, label, unit;
    if (solve === 'wip') {
        result = thr * lt;
        label = 'WIP';
        unit = 'units';
    } else if (solve === 'throughput') {
        result = wip / lt;
        label = 'Throughput';
        unit = 'units/hr';
    } else {
        result = wip / thr;
        label = 'Lead Time';
        unit = 'hours';
    }

    document.getElementById('littles-result-label').textContent = label;
    document.getElementById('littles-result').innerHTML = `${result.toFixed(1)}<span class="calc-result-unit">${unit}</span>`;

    // Update derivation
    let derivation = '';
    if (solve === 'wip') {
        derivation = `
            <div class="step">
                <div class="step-num">Little's Law: Solve for WIP</div>
                <span class="formula">L = λ × W</span><br>
                WIP = Throughput × Lead Time<br>
                WIP = ${thr} × ${lt} = <strong>${result.toFixed(1)} units</strong>
            </div>`;
    } else if (solve === 'throughput') {
        derivation = `
            <div class="step">
                <div class="step-num">Little's Law: Solve for Throughput</div>
                <span class="formula">λ = L ÷ W</span><br>
                Throughput = WIP ÷ Lead Time<br>
                Throughput = ${wip} ÷ ${lt} = <strong>${result.toFixed(1)} units/hr</strong>
            </div>`;
    } else {
        derivation = `
            <div class="step">
                <div class="step-num">Little's Law: Solve for Lead Time</div>
                <span class="formula">W = L ÷ λ</span><br>
                Lead Time = WIP ÷ Throughput<br>
                Lead Time = ${wip} ÷ ${thr} = <strong>${result.toFixed(1)} hours</strong>
            </div>`;
    }
    derivation += `
        <div class="step">
            <div class="step-num">Interpretation</div>
            Little's Law (L = λW) is a fundamental relationship that holds for <em>any</em> stable system, regardless of arrival distribution or service pattern.
        </div>`;
    document.getElementById('littles-derivation-body').innerHTML = derivation;

    // Publish to shared state
    SvendOps.publish('littlesResult', parseFloat(result.toFixed(1)), unit, "Little's Law");

    // Little's Law bar chart — show all 3 variables
    const actualWip = solve === 'wip' ? result : wip;
    const actualThr = solve === 'throughput' ? result : thr;
    const actualLt = solve === 'leadtime' ? result : lt;
    ForgeViz.render(document.getElementById('littles-chart'), {
        title: `L = λ × W → ${actualWip.toFixed(1)} = ${actualThr.toFixed(1)} × ${actualLt.toFixed(1)}`,
        chart_type: 'bar',
        traces: [{ x: ['WIP (L)', 'Throughput (λ)', 'Lead Time (W)'],
            y: [actualWip, actualThr, actualLt],
            name: '', trace_type: 'bar', color: ['#3498db', '#27ae60', '#f39c12'] }],
        reference_lines: [], zones: [], markers: [],
        y_axis: { label: '' }, x_axis: { label: '' }
    });
}

// ============================================================================
// Pitch
// ============================================================================

function calcPitch() {
    const takt = parseFloat(document.getElementById('pitch-takt').value) || 0;
    const pack = parseFloat(document.getElementById('pitch-pack').value) || 1;

    const pitchSec = takt * pack;
    const pitchMin = pitchSec / 60;
    const perHour = 60 / pitchMin;
    const perShift = perHour * 8;

    document.getElementById('pitch-result').innerHTML = `${pitchMin.toFixed(1)}<span class="calc-result-unit">min</span>`;
    document.getElementById('pitch-per-hour').textContent = perHour.toFixed(1);
    document.getElementById('pitch-per-shift').textContent = Math.round(perShift);

    // Update derivation
    document.getElementById('pitch-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Calculate Pitch</div>
            <span class="formula">Pitch = Takt Time × Pack-out Quantity</span><br>
            Pitch = ${takt} sec × ${pack} units = ${pitchSec} sec = <strong>${pitchMin.toFixed(1)} min</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 2: Pickups per Hour</div>
            <span class="formula">Per Hour = 60 ÷ Pitch (min)</span><br>
            = 60 ÷ ${pitchMin.toFixed(1)} = <strong>${perHour.toFixed(1)} pickups/hr</strong>
        </div>
        <div class="step">
            <div class="step-num">Interpretation</div>
            The water spider visits every <strong>${pitchMin.toFixed(1)} minutes</strong> to collect a container of ${pack} units, making <strong>${Math.round(perShift)} pickups per 8-hour shift</strong>.
        </div>
    `;

    SvendOps.publish('pitch', parseFloat(pitchMin.toFixed(1)), 'min', 'Pitch');
}

// ============================================================================
// Inventory Turns
// ============================================================================

function calcTurns() {
    const cogs = parseFloat(document.getElementById('turns-cogs').value) || 0;
    const inv = parseFloat(document.getElementById('turns-inv').value) || 1;

    const turns = cogs / inv;
    const doh = 365 / turns;
    const woh = 52 / turns;

    document.getElementById('turns-result').innerHTML = `${turns.toFixed(1)}<span class="calc-result-unit">/yr</span>`;
    document.getElementById('turns-doh').innerHTML = `${doh.toFixed(1)}<span class="calc-result-unit">days</span>`;
    document.getElementById('turns-woh').innerHTML = `${woh.toFixed(1)}<span class="calc-result-unit">weeks</span>`;

    // Update derivation
    document.getElementById('turns-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Inventory Turns</div>
            <span class="formula">Turns = COGS ÷ Average Inventory</span><br>
            = $${cogs.toLocaleString()} ÷ $${inv.toLocaleString()} = <strong>${turns.toFixed(1)} turns/year</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 2: Days on Hand</div>
            <span class="formula">DOH = 365 ÷ Turns</span><br>
            = 365 ÷ ${turns.toFixed(1)} = <strong>${doh.toFixed(1)} days</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 3: Weeks on Hand</div>
            <span class="formula">WOH = 52 ÷ Turns</span><br>
            = 52 ÷ ${turns.toFixed(1)} = <strong>${woh.toFixed(1)} weeks</strong>
        </div>
    `;

    // Turns gauge chart
    const turnsMax = Math.max(turns * 2, 24);
    ForgeViz.render(document.getElementById('turns-chart'), {
        title: 'Mfg avg: 6-8 | Retail: 8-12 | World class: 20+', subtitle: '',
        chart_type: 'gauge',
        gauge: { value: turns, min: 0, max: turnsMax, label: '/yr' },
        zones: [
            { low: 0, high: 4, axis: 'y', color: 'rgba(231,76,60,0.35)', label: 'Low' },
            { low: 4, high: 8, axis: 'y', color: 'rgba(243,156,18,0.3)', label: 'Avg' },
            { low: 8, high: 12, axis: 'y', color: 'rgba(243,156,18,0.2)', label: 'Good' },
            { low: 12, high: turnsMax, axis: 'y', color: 'rgba(39,174,96,0.35)', label: 'Excellent' }
        ],
        traces: [], reference_lines: [], markers: [],
        x_axis: { label: '' }, y_axis: { label: '' }
    });

    SvendOps.publish('turns', parseFloat(turns.toFixed(1)), '/yr', 'Inv Turns');
    SvendOps.publish('daysOnHand', parseFloat(doh.toFixed(1)), 'days', 'Inv Turns');
}

// ============================================================================
// Cost of Quality
// ============================================================================

function calcCOQ() {
    const training = parseFloat(document.getElementById('coq-training').value) || 0;
    const planning = parseFloat(document.getElementById('coq-planning').value) || 0;
    const systems = parseFloat(document.getElementById('coq-systems').value) || 0;
    const inspection = parseFloat(document.getElementById('coq-inspection').value) || 0;
    const testing = parseFloat(document.getElementById('coq-testing').value) || 0;
    const audits = parseFloat(document.getElementById('coq-audits').value) || 0;
    const internal = parseFloat(document.getElementById('coq-internal').value) || 0;
    const external = parseFloat(document.getElementById('coq-external').value) || 0;

    const prevention = training + planning + systems;
    const appraisal = inspection + testing + audits;
    const failure = internal + external;
    const total = prevention + appraisal + failure;

    const fmt = (n) => n >= 1000000 ? `$${(n/1000000).toFixed(1)}M` : `$${(n/1000).toFixed(0)}K`;

    document.getElementById('coq-prev-total').textContent = fmt(prevention);
    document.getElementById('coq-appr-total').textContent = fmt(appraisal);
    document.getElementById('coq-fail-total').textContent = fmt(failure);
    document.getElementById('coq-total').textContent = fmt(total);

    // Update derivation
    const prevPct = total > 0 ? ((prevention / total) * 100).toFixed(0) : 0;
    const apprPct = total > 0 ? ((appraisal / total) * 100).toFixed(0) : 0;
    const failPct = total > 0 ? ((failure / total) * 100).toFixed(0) : 0;
    document.getElementById('coq-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Prevention Costs</div>
            <span class="formula">Prevention = Training + Planning + Systems</span><br>
            = ${fmt(training)} + ${fmt(planning)} + ${fmt(systems)} = <strong>${fmt(prevention)}</strong> (${prevPct}%)
        </div>
        <div class="step">
            <div class="step-num">Step 2: Appraisal Costs</div>
            <span class="formula">Appraisal = Inspection + Testing + Audits</span><br>
            = ${fmt(inspection)} + ${fmt(testing)} + ${fmt(audits)} = <strong>${fmt(appraisal)}</strong> (${apprPct}%)
        </div>
        <div class="step">
            <div class="step-num">Step 3: Failure Costs</div>
            <span class="formula">Failure = Internal + External</span><br>
            = ${fmt(internal)} + ${fmt(external)} = <strong>${fmt(failure)}</strong> (${failPct}%)
        </div>
        <div class="step">
            <div class="step-num">Step 4: Total Cost of Quality</div>
            <span class="formula">CoQ = Prevention + Appraisal + Failure</span><br>
            = ${fmt(prevention)} + ${fmt(appraisal)} + ${fmt(failure)} = <strong>${fmt(total)}</strong>
        </div>
        <div class="step">
            <div class="step-num">Interpretation</div>
            ${failure > prevention + appraisal ? 'Failure costs dominate — invest more in prevention to reduce total CoQ.' : 'Good balance — prevention spending is keeping failure costs in check.'}
        </div>
    `;

    // Pie chart
    ForgeViz.render(document.getElementById('coq-chart'), {
        title: '', chart_type: 'pie',
        traces: [{ type: 'pie',
            labels: ['Prevention', 'Appraisal', 'Internal Failure', 'External Failure'],
            values: [prevention, appraisal, internal, external],
            colors: ['#4a9f6e', '#3a7f8f', '#e89547', '#e74c3c']
        }],
        reference_lines: [], zones: [], markers: [],
        x_axis: { label: '' }, y_axis: { label: '' }
    });

    SvendOps.publish('coqTotal', total, '$', 'COQ');
    SvendOps.publish('coqFailure', failure, '$', 'COQ');
}

// ============================================================================
// Line Efficiency
// ============================================================================

function calcLineEff() {
    const theoretical = parseFloat(document.getElementById('lineeff-theoretical').value) || 1;
    const scheduled = parseFloat(document.getElementById('lineeff-scheduled').value) || 1;
    const actual = parseFloat(document.getElementById('lineeff-actual').value) || 0;
    const planned = parseFloat(document.getElementById('lineeff-planned').value) || 0;
    const unplanned = parseFloat(document.getElementById('lineeff-unplanned').value) || 0;
    const changeover = parseFloat(document.getElementById('lineeff-changeover').value) || 0;
    const minor = parseFloat(document.getElementById('lineeff-minor').value) || 0;

    const theoreticalOutput = theoretical * (scheduled / 60);
    const efficiency = (actual / theoreticalOutput) * 100;
    const actualRate = actual / (scheduled / 60);
    const lost = theoreticalOutput - actual;

    document.getElementById('lineeff-result').innerHTML = `${efficiency.toFixed(1)}<span class="calc-result-unit">%</span>`;
    document.getElementById('lineeff-rate').innerHTML = `${actualRate.toFixed(0)}<span class="calc-result-unit">/hr</span>`;
    document.getElementById('lineeff-lost').textContent = Math.round(lost);

    // Update derivation
    const running = scheduled - planned - unplanned - changeover - minor;
    document.getElementById('lineeff-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Theoretical Output</div>
            <span class="formula">Theoretical = Rate × (Scheduled ÷ 60)</span><br>
            = ${theoretical} × (${scheduled} ÷ 60) = <strong>${theoreticalOutput.toFixed(0)} units</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 2: Running Time</div>
            <span class="formula">Running = Scheduled − Planned − Unplanned − Changeover − Minor</span><br>
            = ${scheduled} − ${planned} − ${unplanned} − ${changeover} − ${minor} = <strong>${running} min</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 3: Line Efficiency</div>
            <span class="formula">Efficiency = (Actual Output ÷ Theoretical) × 100</span><br>
            = (${actual} ÷ ${theoreticalOutput.toFixed(0)}) × 100 = <strong>${efficiency.toFixed(1)}%</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 4: Lost Units</div>
            <span class="formula">Lost = Theoretical − Actual</span><br>
            = ${theoreticalOutput.toFixed(0)} − ${actual} = <strong>${Math.round(lost)} units</strong>
        </div>
    `;

    // Waterfall chart
    const categories = ['Scheduled', 'Planned', 'Unplanned', 'Changeover', 'Minor Stops', 'Running'];
    const values = [scheduled, -planned, -unplanned, -changeover, -minor, 0];

    let cumulative = scheduled;
    const measures = ['absolute', 'relative', 'relative', 'relative', 'relative', 'total'];

    // Waterfall as stacked bar (positive=green, negative=red, total=blue)
    const waterfallColors = values.map((v, i) => i === 0 || i === 5 ? '#3a7f8f' : '#e74c3c');
    ForgeViz.render(document.getElementById('lineeff-chart'), {
        title: '', chart_type: 'bar',
        traces: [{ x: categories,
            y: values.map((v, i) => i === 5 ? running : Math.abs(v)),
            name: '', trace_type: 'bar', color: waterfallColors }],
        reference_lines: [], zones: [], markers: [],
        y_axis: { label: 'Minutes' }, x_axis: { label: '' }
    });

    SvendOps.publish('lineEffCalc', parseFloat(efficiency.toFixed(1)), '%', 'Line Eff');
    SvendOps.publish('lineEffActualRate', parseFloat(actualRate.toFixed(0)), '/hr', 'Line Eff');
}

// ============================================================================
// OLE (Overall Labor Effectiveness)
// ============================================================================

function calcOLE() {
    const scheduled = parseFloat(document.getElementById('ole-scheduled').value) || 1;
    const absent = parseFloat(document.getElementById('ole-absent').value) || 0;
    const breaks = parseFloat(document.getElementById('ole-breaks').value) || 0;
    const standard = parseFloat(document.getElementById('ole-standard').value) || 1;
    const actual = parseFloat(document.getElementById('ole-actual').value) || 0;
    const good = parseFloat(document.getElementById('ole-good').value) || 0;

    const available = scheduled - absent - breaks;
    const availability = available / scheduled;
    const expectedOutput = available * standard;
    const performance = Math.min(1, actual / expectedOutput);
    const quality = actual > 0 ? good / actual : 0;
    const ole = availability * performance * quality * 100;

    const oleEl = document.getElementById('ole-result');
    oleEl.textContent = `${ole.toFixed(1)}%`;
    oleEl.className = 'oee-value ' + (ole >= 85 ? 'oee-world-class' : ole >= 65 ? 'oee-good' : 'oee-poor');

    document.getElementById('ole-availability').textContent = `${(availability * 100).toFixed(1)}%`;
    document.getElementById('ole-performance').textContent = `${(performance * 100).toFixed(1)}%`;
    document.getElementById('ole-quality').textContent = `${(quality * 100).toFixed(1)}%`;

    // Breakdown stats
    document.getElementById('ole-avail-hours').textContent = `${available.toFixed(1)} hrs`;
    document.getElementById('ole-good-display').textContent = `${good} units`;

    // Update derivation
    document.getElementById('ole-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Availability</div>
            <span class="formula">Availability = (Scheduled − Absent − Breaks) ÷ Scheduled</span><br>
            = (${scheduled} − ${absent} − ${breaks}) ÷ ${scheduled} = ${available.toFixed(1)} ÷ ${scheduled} = <strong>${(availability * 100).toFixed(1)}%</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 2: Performance</div>
            <span class="formula">Performance = Actual Output ÷ (Available × Standard Rate)</span><br>
            = ${actual} ÷ (${available.toFixed(1)} × ${standard}) = ${actual} ÷ ${expectedOutput.toFixed(0)} = <strong>${(performance * 100).toFixed(1)}%</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 3: Quality</div>
            <span class="formula">Quality = Good Units ÷ Actual Output</span><br>
            = ${good} ÷ ${actual} = <strong>${(quality * 100).toFixed(1)}%</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 4: OLE</div>
            <span class="formula">OLE = Availability × Performance × Quality</span><br>
            = ${(availability * 100).toFixed(1)}% × ${(performance * 100).toFixed(1)}% × ${(quality * 100).toFixed(1)}% = <strong>${ole.toFixed(1)}%</strong>
        </div>
    `;

    // Donut chart
    const availLoss = (1 - availability) * 100;
    const perfLoss = availability * (1 - performance) * 100;
    const qualLoss = availability * performance * (1 - quality) * 100;

    const oleColor = ole >= 85 ? '#27ae60' : ole >= 65 ? '#f39c12' : '#e74c3c';
    ForgeViz.render(document.getElementById('ole-chart'), {
        title: '', chart_type: 'donut',
        traces: [{ type: 'donut',
            labels: ['OLE', 'Availability Loss', 'Performance Loss', 'Quality Loss'],
            values: [ole, availLoss, perfLoss, qualLoss],
            colors: [oleColor, '#e74c3c', '#f39c12', '#9b59b6'],
            center_label: `${ole.toFixed(0)}%`
        }],
        reference_lines: [], zones: [], markers: [],
        x_axis: { label: '' }, y_axis: { label: '' }
    });

    SvendOps.publish('ole', parseFloat(ole.toFixed(1)), '%', 'OLE');
}

// ============================================================================
// Cycle Time Study
// ============================================================================

let cycleData = [
    { name: 'Pick up part', time: 5, type: 'nva' },
    { name: 'Position in fixture', time: 8, type: 'nva' },
    { name: 'Machine cycle', time: 45, type: 'va' },
    { name: 'Wait for next part', time: 12, type: 'wait' },
    { name: 'Inspect', time: 10, type: 'va' },
    { name: 'Set aside', time: 5, type: 'nva' },
];

function renderCycleInputs() {
    const container = document.getElementById('cycletime-elements');
    container.innerHTML = cycleData.map((c, i) => `
        <div class="yamazumi-station" style="gap: 8px;">
            <input type="text" value="${c.name}" style="flex:1; padding: 8px;" oninput="updateCycle(${i}, 'name', this.value)">
            <input type="number" value="${c.time}" style="width:70px; text-align:right;" oninput="updateCycle(${i}, 'time', this.value)">
            <span style="color: var(--text-dim); font-size: 12px;">sec</span>
            <select style="width: 80px; padding: 8px;" onchange="updateCycle(${i}, 'type', this.value)">
                <option value="va" ${c.type === 'va' ? 'selected' : ''}>VA</option>
                <option value="nva" ${c.type === 'nva' ? 'selected' : ''}>NVA</option>
                <option value="wait" ${c.type === 'wait' ? 'selected' : ''}>Wait</option>
            </select>
            <button class="yamazumi-station-remove" onclick="removeCycle(${i})">&times;</button>
        </div>
    `).join('');
    calcCycle();
}

function addCycleElement() {
    cycleData.push({ name: 'New element', time: 10, type: 'nva' });
    renderCycleInputs();
}

function removeCycle(idx) {
    cycleData.splice(idx, 1);
    renderCycleInputs();
}

function updateCycle(idx, field, value) {
    if (field === 'time') value = parseFloat(value) || 0;
    cycleData[idx][field] = value;
    calcCycle();
}

function calcCycle() {
    const va = cycleData.filter(c => c.type === 'va').reduce((a, c) => a + c.time, 0);
    const nva = cycleData.filter(c => c.type === 'nva').reduce((a, c) => a + c.time, 0);
    const wait = cycleData.filter(c => c.type === 'wait').reduce((a, c) => a + c.time, 0);
    const total = va + nva + wait;

    document.getElementById('cycle-total').innerHTML = `${total}<span class="calc-result-unit">sec</span>`;
    document.getElementById('cycle-va').innerHTML = `${va}<span class="calc-result-unit">sec</span>`;
    document.getElementById('cycle-nva').innerHTML = `${nva}<span class="calc-result-unit">sec</span>`;
    document.getElementById('cycle-wait').innerHTML = `${wait}<span class="calc-result-unit">sec</span>`;
    document.getElementById('cycle-va-pct').textContent = total > 0 ? `${((va / total) * 100).toFixed(0)}%` : '0%';
    document.getElementById('cycle-waste-pct').textContent = total > 0 ? `${(((nva + wait) / total) * 100).toFixed(0)}%` : '0%';

    // Update derivation
    const vaPct = total > 0 ? ((va / total) * 100).toFixed(0) : 0;
    const nvaPct = total > 0 ? ((nva / total) * 100).toFixed(0) : 0;
    const waitPct = total > 0 ? ((wait / total) * 100).toFixed(0) : 0;
    document.getElementById('cycletime-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Value-Add Time</div>
            VA = ${cycleData.filter(c => c.type === 'va').map(c => c.time).join(' + ') || '0'} = <strong>${va} sec</strong> (${vaPct}%)
        </div>
        <div class="step">
            <div class="step-num">Step 2: Non Value-Add Time</div>
            NVA = ${cycleData.filter(c => c.type === 'nva').map(c => c.time).join(' + ') || '0'} = <strong>${nva} sec</strong> (${nvaPct}%)
        </div>
        <div class="step">
            <div class="step-num">Step 3: Wait Time</div>
            Wait = ${cycleData.filter(c => c.type === 'wait').map(c => c.time).join(' + ') || '0'} = <strong>${wait} sec</strong> (${waitPct}%)
        </div>
        <div class="step">
            <div class="step-num">Step 4: Total Cycle Time</div>
            <span class="formula">Total = VA + NVA + Wait</span><br>
            = ${va} + ${nva} + ${wait} = <strong>${total} sec</strong>
        </div>
    `;

    ForgeViz.render(document.getElementById('cycletime-chart'), {
        title: '', chart_type: 'pie',
        traces: [{ type: 'pie',
            labels: ['Value-Add', 'Non Value-Add', 'Wait'],
            values: [va, nva, wait],
            colors: ['#4a9f6e', '#f39c12', '#e74c3c']
        }],
        reference_lines: [], zones: [], markers: [],
        x_axis: { label: '' }, y_axis: { label: '' }
    });

    SvendOps.publish('cycleTimeTotal', total, 'sec', 'Cycle Time');
    SvendOps.publish('cycleTimeVA', va, 'sec', 'Cycle Time');
}

// ============================================================================
// Before / After
// ============================================================================

let baData = [
    { metric: 'Cycle Time', before: 120, after: 85, unit: 'sec', lowerBetter: true },
    { metric: 'Defect Rate', before: 3.5, after: 1.2, unit: '%', lowerBetter: true },
    { metric: 'OEE', before: 65, after: 82, unit: '%', lowerBetter: false },
];

function renderBAInputs() {
    const container = document.getElementById('beforeafter-metrics');
    container.innerHTML = baData.map((m, i) => `
        <div style="display: grid; grid-template-columns: 1fr 100px 100px 60px 80px 30px; gap: 8px; align-items: center; padding: 12px; background: var(--bg-secondary); border-radius: 8px;">
            <input type="text" value="${m.metric}" placeholder="Metric" style="padding: 8px;" oninput="updateBA(${i}, 'metric', this.value)">
            <input type="number" value="${m.before}" placeholder="Before" style="text-align: right; padding: 8px;" oninput="updateBA(${i}, 'before', this.value)">
            <input type="number" value="${m.after}" placeholder="After" style="text-align: right; padding: 8px;" oninput="updateBA(${i}, 'after', this.value)">
            <input type="text" value="${m.unit}" placeholder="Unit" style="padding: 8px;" oninput="updateBA(${i}, 'unit', this.value)">
            <select style="padding: 8px;" onchange="updateBA(${i}, 'lowerBetter', this.value === 'true')">
                <option value="true" ${m.lowerBetter ? 'selected' : ''}>↓ Better</option>
                <option value="false" ${!m.lowerBetter ? 'selected' : ''}>↑ Better</option>
            </select>
            <button class="yamazumi-station-remove" onclick="removeBA(${i})">&times;</button>
        </div>
    `).join('');
    calcBA();
}

function addBAMetric() {
    baData.push({ metric: 'New Metric', before: 100, after: 80, unit: '', lowerBetter: true });
    renderBAInputs();
}

function removeBA(idx) {
    baData.splice(idx, 1);
    renderBAInputs();
}

async function pullVSMCompareIntoBA() {
    try {
        const resp = await fetch('/api/vsm/', { credentials: 'same-origin' });
        const data = await resp.json();
        const maps = data.maps || [];
        const currentMaps = maps.filter(m => m.status === 'current');
        const futureMaps = maps.filter(m => m.status === 'future');
        if (currentMaps.length === 0) { showToast('No current-state VSM found', 'warning'); return; }
        if (futureMaps.length === 0) { showToast('No future-state VSM found. Create a Future State map to compare.', 'warning'); return; }

        // Fetch both VSMs
        const [curResp, futResp] = await Promise.all([
            fetch(`/api/vsm/${currentMaps[0].id}/`, { credentials: 'same-origin' }),
            fetch(`/api/vsm/${futureMaps[0].id}/`, { credentials: 'same-origin' })
        ]);
        const curData = await curResp.json();
        const futData = await futResp.json();
        const cur = curData.vsm;
        const fut = futData.vsm;

        // Build comparison metrics
        const newData = [];
        if (cur.total_lead_time != null && fut.total_lead_time != null) {
            newData.push({ metric: 'Lead Time', before: parseFloat(cur.total_lead_time.toFixed(3)), after: parseFloat(fut.total_lead_time.toFixed(3)), unit: 'days', lowerBetter: true });
        }
        if (cur.total_process_time != null && fut.total_process_time != null) {
            newData.push({ metric: 'Process Time', before: Math.round(cur.total_process_time), after: Math.round(fut.total_process_time), unit: 'sec', lowerBetter: true });
        }
        if (cur.pce != null && fut.pce != null) {
            newData.push({ metric: 'PCE', before: parseFloat(cur.pce.toFixed(2)), after: parseFloat(fut.pce.toFixed(2)), unit: '%', lowerBetter: false });
        }

        // Per-step CT deltas for matching step names
        const curSteps = cur.process_steps || [];
        const futSteps = fut.process_steps || [];
        for (const cs of curSteps) {
            const fs = futSteps.find(f => f.name && f.name === cs.name);
            if (fs && cs.cycle_time && fs.cycle_time) {
                newData.push({ metric: `${cs.name} C/T`, before: cs.cycle_time, after: fs.cycle_time, unit: 'sec', lowerBetter: true });
            }
        }

        if (newData.length === 0) { showToast('No comparable metrics found between current and future VSMs', 'warning'); return; }
        baData = newData;
        renderBAInputs();
        showToast(`Loaded ${newData.length} metrics from VSM comparison`);
    } catch (e) {
        showToast('Failed to pull VSM data: ' + safeStr(e, 'Unknown error'), 'error');
    }
}

function updateBA(idx, field, value) {
    if (field === 'before' || field === 'after') value = parseFloat(value) || 0;
    baData[idx][field] = value;
    calcBA();
}

function calcBA() {
    if (baData.length === 0) return;

    const improvements = baData.map(m => {
        const change = m.after - m.before;
        const pctChange = m.before !== 0 ? (change / m.before) * 100 : 0;
        const improved = m.lowerBetter ? change < 0 : change > 0;
        return { ...m, change, pctChange, improved };
    });

    ForgeViz.render(document.getElementById('beforeafter-chart'), {
        title: '', chart_type: 'grouped_bar',
        traces: [
            { x: improvements.map(m => m.metric), y: improvements.map(m => m.before),
              name: 'Before', trace_type: 'bar', color: '#666' },
            { x: improvements.map(m => m.metric), y: improvements.map(m => m.after),
              name: 'After', trace_type: 'bar',
              color: improvements.map(m => m.improved ? '#4a9f6e' : '#e74c3c') }
        ],
        reference_lines: [], zones: [], markers: [],
        y_axis: { label: '' }, x_axis: { label: '' }
    });

    // Show Synara action bar
    const actionBar = document.getElementById('ba-synara-action');
    if (actionBar) actionBar.style.display = 'block';
}

async function logBAToSynara() {
    if (baData.length === 0) { showToast('No before/after data to log', 'warning'); return; }

    const improvements = baData.map(m => {
        const change = m.after - m.before;
        const pctChange = m.before !== 0 ? (change / m.before) * 100 : 0;
        const improved = m.lowerBetter ? change < 0 : change > 0;
        return { ...m, change, pctChange, improved };
    });

    const improvedCount = improvements.filter(m => m.improved).length;
    const details = improvements.map(m =>
        `${m.metric}: ${m.before}${m.unit} \u2192 ${m.after}${m.unit} (${m.pctChange >= 0 ? '+' : ''}${m.pctChange.toFixed(1)}%${m.improved ? ' \u2713' : ' \u2717'})`
    ).join('; ');
    const summary = `Before/After: ${improvedCount}/${baData.length} metrics improved. ${details}`;

    // Try posting to Synara if workbench context available
    const wbId = typeof currentWorkbench !== 'undefined' && currentWorkbench?.id;
    if (wbId) {
        try {
            const hypResp = await fetch(`/api/synara/${wbId}/hypotheses/`);
            const hypData = await hypResp.json();
            const hypotheses = hypData.hypotheses || [];
            if (hypotheses.length === 0) {
                showToast('No hypotheses yet \u2014 create one in the Synara tab first', 'warning');
                return;
            }
            // Post as general evidence (not linked to specific hypothesis)
            await fetch(`/api/synara/${wbId}/evidence/add/`, {
                method: 'POST', credentials: 'same-origin',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ event: summary, strength: improvedCount / baData.length, source: 'before_after' })
            });
            showToast('Evidence logged to Synara');
        } catch (e) {
            showToast('Failed to log evidence: ' + safeStr(e, 'Unknown error'), 'error');
        }
    } else {
        sessionStorage.setItem('pendingEvidence', JSON.stringify({ source: 'Before/After', summary, confidence: improvedCount / baData.length }));
        showToast('Evidence saved \u2014 will be available when you open a workbench');
    }
}

// ============================================================================
// Heijunka
// ============================================================================

let heijunkaData = [
    { product: 'A', demand: 100 },
    { product: 'B', demand: 60 },
    { product: 'C', demand: 40 },
];

function renderHeijunkaInputs() {
    const container = document.getElementById('heijunka-products');
    container.innerHTML = heijunkaData.map((p, i) => `
        <div class="yamazumi-station" style="gap: 8px;">
            <input type="text" value="${p.product}" style="width: 80px; padding: 8px;" oninput="updateHeijunka(${i}, 'product', this.value)">
            <span style="color: var(--text-dim);">Daily demand:</span>
            <input type="number" value="${p.demand}" style="width: 100px; text-align: right; padding: 8px;" oninput="updateHeijunka(${i}, 'demand', this.value)">
            <button class="yamazumi-station-remove" onclick="removeHeijunka(${i})">&times;</button>
        </div>
    `).join('');
    calcHeijunka();
}

function addHeijunkaProduct() {
    heijunkaData.push({ product: String.fromCharCode(65 + heijunkaData.length), demand: 50 });
    renderHeijunkaInputs();
}

function removeHeijunka(idx) {
    heijunkaData.splice(idx, 1);
    renderHeijunkaInputs();
}

function updateHeijunka(idx, field, value) {
    if (field === 'demand') value = parseInt(value) || 0;
    heijunkaData[idx][field] = value;
    calcHeijunka();
}

function calcHeijunka() {
    const time = parseFloat(document.getElementById('heijunka-time').value) || 480;
    const pitch = parseFloat(document.getElementById('heijunka-pitch').value) || 20;

    const intervals = Math.floor(time / pitch);
    const totalDemand = heijunkaData.reduce((a, p) => a + p.demand, 0);
    const perPitch = totalDemand / intervals;

    document.getElementById('heijunka-intervals').textContent = intervals;
    document.getElementById('heijunka-total').textContent = totalDemand;
    document.getElementById('heijunka-per-pitch').textContent = perPitch.toFixed(1);

    // Update derivation
    const productBreakdown = heijunkaData.map(p =>
        `${p.product}: ${p.demand} units (${totalDemand > 0 ? ((p.demand / totalDemand) * 100).toFixed(0) : 0}% of mix)`
    ).join('<br>');
    document.getElementById('heijunka-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Calculate Intervals</div>
            <span class="formula">Intervals = Available Time ÷ Pitch</span><br>
            = ${time} ÷ ${pitch} = <strong>${intervals} intervals</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 2: Total Demand</div>
            ${productBreakdown}<br>
            Total = <strong>${totalDemand} units</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 3: Units per Pitch Interval</div>
            <span class="formula">Per Pitch = Total Demand ÷ Intervals</span><br>
            = ${totalDemand} ÷ ${intervals} = <strong>${perPitch.toFixed(1)} units/interval</strong>
        </div>
    `;

    // Calculate pattern
    const gcd = (a, b) => b === 0 ? a : gcd(b, a % b);
    const demands = heijunkaData.map(p => p.demand);
    const patternGcd = demands.reduce((a, b) => gcd(a, b));
    const pattern = heijunkaData.map(p => p.product.repeat(p.demand / patternGcd)).join('');
    document.getElementById('heijunka-pattern').textContent = pattern.length > 20
        ? `${pattern.substring(0, 20)}... (repeats ${intervals / (totalDemand / patternGcd)}×)`
        : `${pattern} (repeats ${Math.floor(intervals / (pattern.length || 1))}×)`;

    // Heijunka box visualization (first 10 intervals)
    const boxData = [];
    let patternIdx = 0;
    for (let i = 0; i < Math.min(10, intervals); i++) {
        const slot = [];
        for (let j = 0; j < Math.ceil(perPitch); j++) {
            if (patternIdx < pattern.length) {
                slot.push(pattern[patternIdx % pattern.length]);
                patternIdx++;
            }
        }
        boxData.push(slot);
    }

    // Create heatmap-style visualization
    const colors = {};
    heijunkaData.forEach((p, i) => {
        const hue = (i * 137) % 360;
        colors[p.product] = `hsl(${hue}, 60%, 50%)`;
    });

    const traces = heijunkaData.map(p => ({
        x: Array.from({ length: Math.min(10, intervals) }, (_, i) => `Pitch ${i + 1}`),
        y: boxData.map(slot => slot.filter(s => s === p.product).length),
        type: 'bar',
        name: p.product,
        marker: { color: colors[p.product] }
    }));

    const fvTraces = heijunkaData.map(p => ({
        x: Array.from({ length: Math.min(10, intervals) }, (_, i) => `Pitch ${i + 1}`),
        y: boxData.map(slot => slot.filter(s => s === p.product).length),
        name: p.product, trace_type: 'bar', color: colors[p.product]
    }));
    ForgeViz.render(document.getElementById('heijunka-chart'), {
        title: '', chart_type: 'stacked_bar',
        traces: fvTraces,
        reference_lines: [], zones: [], markers: [],
        y_axis: { label: 'Units' }, x_axis: { label: '' }
    });
}
