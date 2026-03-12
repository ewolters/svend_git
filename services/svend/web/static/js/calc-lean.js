/**
 * calc-lean.js — Lean Method Calculators (SMED, PFA, WFA)
 *
 * Load order: after calc-core.js (uses SvendOps, renderNextSteps, showToast)
 * Extracted from: calculators.html (inline script)
 *
 * Provides: SMED analysis with baseline comparison and conversion suggestions,
 * changeover matrix, Product Flow Analysis (TIPS), Workflow Analysis (therbligs).
 */

// ============================================================================
// SMED Analysis
// ============================================================================

let smedData = [
    { name: 'Get tools', time: 5, type: 'external' },
    { name: 'Remove old die', time: 8, type: 'internal' },
    { name: 'Install new die', time: 10, type: 'internal' },
    { name: 'Adjust settings', time: 7, type: 'internal' },
    { name: 'First piece check', time: 5, type: 'internal' },
    { name: 'Cleanup', time: 10, type: 'external' },
];

let smedBaseline = null; // Stores baseline snapshot for before/after comparison

function renderSMEDInputs() {
    const container = document.getElementById('smed-elements');
    container.innerHTML = smedData.map((s, i) => `
        <div class="yamazumi-station" style="gap: 8px;">
            <input type="text" value="${s.name}" style="flex:1; padding: 8px;" oninput="updateSMED(${i}, 'name', this.value)">
            <input type="number" value="${s.time}" style="width:70px; text-align:right;" oninput="updateSMED(${i}, 'time', this.value)">
            <span style="color: var(--text-dim); font-size: 12px;">min</span>
            <select style="width: 100px; padding: 8px;" onchange="updateSMED(${i}, 'type', this.value)">
                <option value="internal" ${s.type === 'internal' ? 'selected' : ''}>Internal</option>
                <option value="external" ${s.type === 'external' ? 'selected' : ''}>External</option>
            </select>
            <button class="yamazumi-station-remove" onclick="removeSMED(${i})">&times;</button>
        </div>
    `).join('');
    calcSMED();
}

function addSMEDElement() {
    smedData.push({ name: 'New element', time: 5, type: 'internal' });
    renderSMEDInputs();
}

function removeSMED(idx) {
    smedData.splice(idx, 1);
    renderSMEDInputs();
}

function updateSMED(idx, field, value) {
    if (field === 'time') value = parseFloat(value) || 0;
    smedData[idx][field] = value;
    calcSMED();
}

function calcSMED() {
    const internal = smedData.filter(s => s.type === 'internal').reduce((a, s) => a + s.time, 0);
    const external = smedData.filter(s => s.type === 'external').reduce((a, s) => a + s.time, 0);
    const total = internal + external;
    const reduction = total > 0 ? ((external / total) * 100) : 0;

    document.getElementById('smed-current').innerHTML = `${total}<span class="calc-result-unit">min</span>`;
    document.getElementById('smed-internal').innerHTML = `${internal}<span class="calc-result-unit">min</span>`;
    document.getElementById('smed-external').innerHTML = `${external}<span class="calc-result-unit">min</span>`;
    document.getElementById('smed-target').innerHTML = `${internal}<span class="calc-result-unit">min</span>`;
    document.getElementById('smed-reduction').textContent = `${reduction.toFixed(0)}% (${external} min saved)`;

    // Update derivation
    const intElements = smedData.filter(s => s.type === 'internal');
    const extElements = smedData.filter(s => s.type === 'external');
    document.getElementById('smed-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Classify Elements</div>
            <strong>Internal</strong> (machine stopped): ${intElements.length} elements = ${internal} min<br>
            <strong>External</strong> (machine running): ${extElements.length} elements = ${external} min
        </div>
        <div class="step">
            <div class="step-num">Step 2: Current Total Changeover</div>
            <span class="formula">Current = Internal + External</span><br>
            = ${internal} + ${external} = <strong>${total} min</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 3: SMED Target</div>
            <span class="formula">Target = Internal Only (externalize the rest)</span><br>
            Target changeover = <strong>${internal} min</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 4: Potential Reduction</div>
            <span class="formula">Savings = External time moved outside changeover</span><br>
            = ${external} min saved = <strong>${reduction.toFixed(0)}% reduction</strong>
        </div>
    `;

    // Update internal seconds display for Line Sim link
    document.getElementById('smed-internal-seconds').textContent = `${internal * 60}s`;

    // Update baseline comparison if exists
    if (smedBaseline) {
        const baselineInternal = smedBaseline.internal;
        const improvement = baselineInternal - internal;
        const pctImprove = baselineInternal > 0 ? ((improvement / baselineInternal) * 100) : 0;

        document.getElementById('smed-baseline-compare').style.display = 'block';
        document.getElementById('smed-baseline-total').textContent = baselineInternal;
        document.getElementById('smed-after-total').textContent = internal;

        if (improvement > 0) {
            document.getElementById('smed-improvement').innerHTML =
                `<span style="color: #4a9f6e;">${pctImprove.toFixed(0)}% reduction (${improvement} min saved)</span>`;
        } else if (improvement < 0) {
            document.getElementById('smed-improvement').innerHTML =
                `<span style="color: #e74c3c;">${Math.abs(improvement)} min added</span>`;
        } else {
            document.getElementById('smed-improvement').innerHTML =
                `<span style="color: var(--text-dim);">No change yet</span>`;
        }
    }

    // Update impact calculations
    calcSMEDImpact();

    // Waterfall chart - each step cascades (using bar with base for full color control)
    const names = smedData.map(s => s.name);
    names.push('Total');

    // Calculate cumulative bases for waterfall effect
    let cumulative = 0;
    const bases = [];
    const heights = [];
    const colors = [];

    smedData.forEach(s => {
        bases.push(cumulative);
        heights.push(s.time);
        colors.push(s.type === 'internal' ? '#e74c3c' : '#4a9f6e');
        cumulative += s.time;
    });

    // Total bar starts at 0
    bases.push(0);
    heights.push(total);
    colors.push('#e8c547'); // Svend Gold

    // Create connector lines as shapes
    const shapes = [];
    for (let i = 0; i < smedData.length; i++) {
        const nextBase = i < smedData.length - 1 ? bases[i + 1] : 0;
        const currentTop = bases[i] + heights[i];
        shapes.push({
            type: 'line',
            x0: i + 0.4, x1: i + 0.6,
            y0: currentTop, y1: currentTop,
            line: { color: 'rgba(150,150,150,0.5)', width: 1, dash: 'dot' }
        });
    }

    Plotly.newPlot('smed-chart', [{
        type: 'bar',
        x: names,
        y: heights,
        base: bases,
        text: heights.map(h => `${h} min`),
        textposition: 'outside',
        textfont: { size: 11, color: '#a0a0a0' },
        marker: { color: colors },
        hovertemplate: '%{x}: %{y} min<extra></extra>'
    }], {
        margin: { t: 40, b: 100, l: 50, r: 20 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        shapes: shapes,
        xaxis: {
            tickangle: -30,
            tickfont: { size: 11, color: '#a0a0a0' },
            gridcolor: 'rgba(150,150,150,0.1)'
        },
        yaxis: {
            title: 'Minutes',
            titlefont: { size: 12, color: '#a0a0a0' },
            tickfont: { size: 11, color: '#a0a0a0' },
            gridcolor: 'rgba(150,150,150,0.1)',
            zeroline: true,
            zerolinecolor: 'rgba(150,150,150,0.3)'
        },
        showlegend: false
    }, { responsive: true, displayModeBar: false });

    // Publish to shared state
    SvendOps.publish('changeoverInternal', internal, 'min', 'SMED');

    renderNextSteps('smed-next-steps', [
        { title: 'Update EPEI', desc: 'Recalculate schedule interval with new C/O', calcId: 'epei', pullKey: 'changeoverInternal', pullTarget: 'epei-changeover' },
        { title: 'Simulate Impact', desc: 'Model the changeover reduction in line sim', calcId: 'linesim' },
    ]);
}

// SMED Baseline capture for before/after comparison
function captureBaseline() {
    const internal = smedData.filter(s => s.type === 'internal').reduce((a, s) => a + s.time, 0);
    const external = smedData.filter(s => s.type === 'external').reduce((a, s) => a + s.time, 0);

    smedBaseline = {
        internal: internal,
        external: external,
        total: internal + external,
        elements: JSON.parse(JSON.stringify(smedData)), // Deep copy
        capturedAt: new Date().toISOString()
    };

    // Show clear button
    document.getElementById('smed-clear-baseline').style.display = 'inline-block';

    // Recalculate to show comparison
    calcSMED();
}

function clearBaseline() {
    smedBaseline = null;
    document.getElementById('smed-baseline-compare').style.display = 'none';
    document.getElementById('smed-clear-baseline').style.display = 'none';
}

// Suggest conversions from internal to external
function suggestConversions() {
    const suggestions = [];
    const container = document.getElementById('smed-suggestions');
    const list = document.getElementById('smed-suggestions-list');

    // Analyze each internal element for conversion opportunities
    smedData.forEach((element, idx) => {
        if (element.type !== 'internal') return;

        const name = element.name.toLowerCase();

        // Pattern-based suggestions (Shingo's methodology)
        if (name.includes('get') || name.includes('fetch') || name.includes('find') || name.includes('search')) {
            suggestions.push({
                element: element.name,
                suggestion: 'Pre-stage: Have materials/tools ready at point of use before stopping',
                impact: 'high'
            });
        }
        else if (name.includes('clean') || name.includes('wipe') || name.includes('clear')) {
            suggestions.push({
                element: element.name,
                suggestion: 'Parallel clean: Assign dedicated person or run cleaning while next job starts',
                impact: 'medium'
            });
        }
        else if (name.includes('adjust') || name.includes('align') || name.includes('set') || name.includes('calibrat')) {
            suggestions.push({
                element: element.name,
                suggestion: 'Pre-set: Use intermediate jigs, standard settings, or quick-adjust mechanisms',
                impact: 'high'
            });
        }
        else if (name.includes('heat') || name.includes('warm') || name.includes('preheat') || name.includes('cool')) {
            suggestions.push({
                element: element.name,
                suggestion: 'Pre-condition: Use external heating/cooling before the switch',
                impact: 'high'
            });
        }
        else if (name.includes('check') || name.includes('inspect') || name.includes('verify') || name.includes('test')) {
            suggestions.push({
                element: element.name,
                suggestion: 'Parallel check: Run verification concurrently with first good parts',
                impact: 'medium'
            });
        }
        else if (name.includes('remove') || name.includes('install') || name.includes('mount') || name.includes('attach')) {
            suggestions.push({
                element: element.name,
                suggestion: 'Quick-change: Use functional clamps, one-turn fasteners, or cassette systems',
                impact: 'high'
            });
        }
        else if (name.includes('document') || name.includes('record') || name.includes('log') || name.includes('paper')) {
            suggestions.push({
                element: element.name,
                suggestion: 'Digitize/eliminate: Automate data capture or simplify records',
                impact: 'low'
            });
        }
        else if (name.includes('wait') || name.includes('queue') || name.includes('hold')) {
            suggestions.push({
                element: element.name,
                suggestion: 'Eliminate: This is pure waste — investigate root cause',
                impact: 'high'
            });
        }
        else {
            // Generic suggestion for unrecognized elements
            suggestions.push({
                element: element.name,
                suggestion: 'Analyze: Can this be done while machine is running? Can it be eliminated entirely?',
                impact: 'medium'
            });
        }
    });

    if (suggestions.length === 0) {
        list.innerHTML = '<div style="color: var(--text-dim); padding: 8px;">All elements are already external. Focus on reducing their duration.</div>';
    } else {
        const impactColors = { high: '#4a9f6e', medium: '#e8c547', low: '#a0a0a0' };
        list.innerHTML = suggestions.map(s => `
            <div style="padding: 10px 0; border-bottom: 1px solid var(--border);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                    <span style="font-weight: 600; color: var(--text-primary);">${s.element}</span>
                    <span style="font-size: 10px; padding: 2px 6px; background: ${impactColors[s.impact]}22; color: ${impactColors[s.impact]}; border-radius: 4px; text-transform: uppercase;">${s.impact} impact</span>
                </div>
                <div style="color: var(--text-secondary);">${s.suggestion}</div>
            </div>
        `).join('');
    }

    container.style.display = 'block';
}

// Calculate annual impact of SMED improvements
function calcSMEDImpact() {
    const internal = smedData.filter(s => s.type === 'internal').reduce((a, s) => a + s.time, 0);
    const changoversPerDay = parseFloat(document.getElementById('smed-changeovers-day').value) || 0;
    const daysPerYear = parseFloat(document.getElementById('smed-days-year').value) || 0;
    const hourlyCost = parseFloat(document.getElementById('smed-hourly-cost').value) || 0;

    // If baseline exists, calculate savings from improvement
    let minutesSaved = 0;
    if (smedBaseline) {
        minutesSaved = smedBaseline.internal - internal;
    } else {
        // Without baseline, show potential if all internal converted to external
        // (This is theoretical maximum)
        minutesSaved = internal; // Assume target is getting to zero internal
    }

    if (minutesSaved < 0) minutesSaved = 0; // Don't show negative savings

    const hoursSavedPerYear = (minutesSaved * changoversPerDay * daysPerYear) / 60;
    const currentInternalHoursPerYear = (internal * changoversPerDay * daysPerYear) / 60;

    // Capacity gain = hours recovered as percentage of shift time
    const shiftHoursPerDay = 8;
    const totalShiftHoursPerYear = shiftHoursPerDay * daysPerYear;
    const capacityGainPct = totalShiftHoursPerYear > 0 ? (hoursSavedPerYear / totalShiftHoursPerYear) * 100 : 0;

    const annualValue = hoursSavedPerYear * hourlyCost;

    // Update display
    if (smedBaseline) {
        document.getElementById('smed-hours-saved').textContent = hoursSavedPerYear.toFixed(0) + 'h';
        document.getElementById('smed-capacity-gain').textContent = capacityGainPct.toFixed(1) + '%';
        document.getElementById('smed-annual-value').textContent = '$' + annualValue.toLocaleString(undefined, {maximumFractionDigits: 0});
    } else {
        // Without baseline, show current loss (opportunity)
        document.getElementById('smed-hours-saved').innerHTML = `<span style="color: var(--text-dim);">${currentInternalHoursPerYear.toFixed(0)}h lost</span>`;
        document.getElementById('smed-capacity-gain').innerHTML = `<span style="color: var(--text-dim);">—</span>`;
        document.getElementById('smed-annual-value').innerHTML = `<span style="color: var(--text-dim);">Capture baseline first</span>`;
    }
}

// Apply SMED internal time to Line Simulator
function applySMEDToLineSim() {
    const internal = smedData.filter(s => s.type === 'internal').reduce((a, s) => a + s.time, 0);
    const internalSeconds = internal * 60;

    // Set the Line Sim changeover time
    const input = document.getElementById('ls-changeover-time');
    if (input) {
        input.value = internalSeconds;
    }

    // Navigate to Line Simulator
    showCalc('linesim');

    // Show a brief toast/notification
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed; bottom: 20px; right: 20px; padding: 12px 20px;
        background: var(--accent); color: white; border-radius: 8px;
        font-size: 13px; font-weight: 500; z-index: 9999;
        animation: fadeInUp 0.3s ease;
    `;
    toast.textContent = `Changeover time set to ${internalSeconds}s (${internal} min internal)`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// ============================================================================
// Changeover Matrix
// ============================================================================

let changeoverMatrix = {};

function renderChangeoverMatrix() {
    const productsStr = document.getElementById('changeover-products').value;
    const products = productsStr.split(',').map(p => p.trim()).filter(p => p);

    if (products.length < 2) return;

    // Initialize matrix if needed
    products.forEach(from => {
        if (!changeoverMatrix[from]) changeoverMatrix[from] = {};
        products.forEach(to => {
            if (from !== to && changeoverMatrix[from][to] === undefined) {
                changeoverMatrix[from][to] = Math.floor(Math.random() * 25) + 5;
            }
        });
    });

    // Render table
    let html = `<table style="border-collapse: collapse; width: 100%;">
        <tr><th style="padding: 8px; border: 1px solid var(--border);">From \\ To</th>`;
    products.forEach(p => {
        html += `<th style="padding: 8px; border: 1px solid var(--border);">${p}</th>`;
    });
    html += '</tr>';

    products.forEach(from => {
        html += `<tr><th style="padding: 8px; border: 1px solid var(--border);">${from}</th>`;
        products.forEach(to => {
            if (from === to) {
                html += `<td style="padding: 8px; border: 1px solid var(--border); background: var(--bg-secondary); text-align: center;">-</td>`;
            } else {
                const val = changeoverMatrix[from]?.[to] || 10;
                html += `<td style="padding: 4px; border: 1px solid var(--border);">
                    <input type="number" value="${val}" style="width: 100%; text-align: center; padding: 4px;"
                           onchange="updateChangeover('${from}', '${to}', this.value)">
                </td>`;
            }
        });
        html += '</tr>';
    });
    html += '</table>';

    document.getElementById('changeover-matrix').innerHTML = html;
    calcChangeover(products);
}

function updateChangeover(from, to, value) {
    if (!changeoverMatrix[from]) changeoverMatrix[from] = {};
    changeoverMatrix[from][to] = parseFloat(value) || 0;
    const products = document.getElementById('changeover-products').value.split(',').map(p => p.trim()).filter(p => p);
    calcChangeover(products);
}

function calcChangeover(products) {
    const times = [];
    products.forEach(from => {
        products.forEach(to => {
            if (from !== to && changeoverMatrix[from]?.[to]) {
                times.push(changeoverMatrix[from][to]);
            }
        });
    });

    if (times.length === 0) return;

    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    const min = Math.min(...times);
    const max = Math.max(...times);

    document.getElementById('changeover-avg').innerHTML = `${avg.toFixed(0)}<span class="calc-result-unit">min</span>`;
    document.getElementById('changeover-min').innerHTML = `${min}<span class="calc-result-unit">min</span>`;
    document.getElementById('changeover-max').innerHTML = `${max}<span class="calc-result-unit">min</span>`;

    // Simple greedy sequence (nearest neighbor)
    const sequence = [products[0]];
    const remaining = new Set(products.slice(1));
    while (remaining.size > 0) {
        const current = sequence[sequence.length - 1];
        let bestNext = null, bestTime = Infinity;
        remaining.forEach(next => {
            const time = changeoverMatrix[current]?.[next] || Infinity;
            if (time < bestTime) { bestTime = time; bestNext = next; }
        });
        if (bestNext) { sequence.push(bestNext); remaining.delete(bestNext); }
        else break;
    }
    document.getElementById('changeover-best').textContent = sequence.join('→');

    // Financial: cost per changeover
    const hourlyCost = parseFloat(document.getElementById('changeover-hourly-cost')?.value) || 0;
    const avgCost = (avg / 60) * hourlyCost;
    document.getElementById('changeover-avg-cost').innerHTML = `$${avgCost.toFixed(0)}`;
    // Best sequence total cost
    let seqTotal = 0;
    for (let i = 0; i < sequence.length - 1; i++) {
        seqTotal += changeoverMatrix[sequence[i]]?.[sequence[i + 1]] || 0;
    }
    const seqCost = (seqTotal / 60) * hourlyCost;
    document.getElementById('changeover-seq-cost').innerHTML = `$${seqCost.toFixed(0)}`;
}

// ============================================================================
// PFA — Product Flow Analysis (TIPS)
// ============================================================================

const PFA_CATEGORIES = {
    'T': { name: 'Transport', color: '#3498db', type: 'NVA' },
    'I': { name: 'Inspect', color: '#9b59b6', type: 'NVA' },
    'P': { name: 'Process', color: '#4a9f6e', type: 'VA' },
    'S-B': { name: 'Storage (Between)', color: '#e67e22', type: 'NVA' },
    'S-L': { name: 'Storage (Lot)', color: '#d35400', type: 'NVA' },
    'S-W': { name: 'Storage (Within)', color: '#e74c3c', type: 'NVA' }
};

let pfaSteps = [];
let pfaBaseline = null;

function renderPFASteps() {
    const container = document.getElementById('pfa-steps');
    if (!container) return;

    container.innerHTML = pfaSteps.map((step, i) => `
        <div style="display: grid; grid-template-columns: 40px 1fr 100px 80px 80px 30px; gap: 8px; align-items: center; padding: 12px; background: var(--bg-secondary); border-radius: 8px;">
            <span style="font-size: 12px; color: var(--text-dim); text-align: center;">${i + 1}</span>
            <input type="text" value="${step.desc}" placeholder="Step description"
                   style="padding: 8px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary);"
                   oninput="updatePFAStep(${i}, 'desc', this.value)">
            <select style="padding: 8px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary);"
                    onchange="updatePFAStep(${i}, 'cat', this.value)">
                ${Object.entries(PFA_CATEGORIES).map(([k, v]) =>
                    `<option value="${k}" ${step.cat === k ? 'selected' : ''} style="color: ${v.color};">${k} - ${v.name}</option>`
                ).join('')}
            </select>
            <div style="display: flex; align-items: center; gap: 4px;">
                <input type="number" value="${step.time}" min="0" step="0.1"
                       style="width: 50px; padding: 6px; text-align: right; background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary);"
                       oninput="updatePFAStep(${i}, 'time', parseFloat(this.value) || 0)">
                <span style="font-size: 11px; color: var(--text-dim);">min</span>
            </div>
            <div style="display: flex; align-items: center; gap: 4px;">
                <input type="number" value="${step.dist || 0}" min="0"
                       style="width: 50px; padding: 6px; text-align: right; background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary);"
                       oninput="updatePFAStep(${i}, 'dist', parseFloat(this.value) || 0)">
                <span style="font-size: 11px; color: var(--text-dim);">m</span>
            </div>
            <button onclick="removePFAStep(${i})" style="padding: 4px 8px; background: transparent; border: none; color: var(--text-dim); cursor: pointer; font-size: 16px;">&times;</button>
        </div>
    `).join('');

    calcPFA();
}

function addPFAStep() {
    pfaSteps.push({ desc: '', cat: 'P', time: 1, dist: 0 });
    renderPFASteps();
}

function removePFAStep(idx) {
    pfaSteps.splice(idx, 1);
    renderPFASteps();
}

function updatePFAStep(idx, field, value) {
    pfaSteps[idx][field] = value;
    calcPFA();
}

function loadPFAExample() {
    pfaSteps = [
        { desc: 'Retrieve from warehouse', cat: 'S-B', time: 0, dist: 0 },
        { desc: 'Transport to staging', cat: 'T', time: 5, dist: 50 },
        { desc: 'Wait for batch', cat: 'S-L', time: 15, dist: 0 },
        { desc: 'Move to machine', cat: 'T', time: 2, dist: 20 },
        { desc: 'Load and machine', cat: 'P', time: 8, dist: 0 },
        { desc: 'Inspect dimensions', cat: 'I', time: 3, dist: 0 },
        { desc: 'Queue for next op', cat: 'S-W', time: 10, dist: 0 },
        { desc: 'Transport to assembly', cat: 'T', time: 4, dist: 35 },
        { desc: 'Wait for kitting', cat: 'S-B', time: 12, dist: 0 },
        { desc: 'Assemble', cat: 'P', time: 15, dist: 0 },
        { desc: 'Final inspection', cat: 'I', time: 5, dist: 0 },
        { desc: 'Pack and label', cat: 'P', time: 3, dist: 0 },
        { desc: 'Transport to shipping', cat: 'T', time: 3, dist: 40 },
    ];
    renderPFASteps();
}

function capturePFABaseline() {
    pfaBaseline = {
        steps: JSON.parse(JSON.stringify(pfaSteps)),
        metrics: calcPFAMetrics(),
        capturedAt: new Date().toISOString()
    };
    document.getElementById('pfa-clear-baseline').style.display = 'inline-block';
    calcPFA();
}

function clearPFABaseline() {
    pfaBaseline = null;
    document.getElementById('pfa-clear-baseline').style.display = 'none';
    document.getElementById('pfa-baseline-compare').style.display = 'none';
    calcPFA();
}

function calcPFAMetrics() {
    const totals = { T: 0, I: 0, P: 0, 'S-B': 0, 'S-L': 0, 'S-W': 0 };
    let totalTime = 0, totalDist = 0;

    pfaSteps.forEach(s => {
        totals[s.cat] = (totals[s.cat] || 0) + s.time;
        totalTime += s.time;
        totalDist += (s.dist || 0);
    });

    const processTime = totals['P'] || 0;
    const storageTime = (totals['S-B'] || 0) + (totals['S-L'] || 0) + (totals['S-W'] || 0);
    const processRatio = totalTime > 0 ? (processTime / totalTime) * 100 : 0;

    return { totals, totalTime, totalDist, processTime, storageTime, processRatio, stepCount: pfaSteps.length };
}

function calcPFA() {
    const m = calcPFAMetrics();

    document.getElementById('pfa-process-ratio').innerHTML = `${m.processRatio.toFixed(0)}<span class="calc-result-unit">%</span>`;
    document.getElementById('pfa-total-time').innerHTML = `${m.totalTime.toFixed(0)}<span class="calc-result-unit">min</span>`;
    document.getElementById('pfa-total-dist').innerHTML = `${m.totalDist.toFixed(0)}<span class="calc-result-unit">m</span>`;
    document.getElementById('pfa-step-count').textContent = m.stepCount;

    // Breakdown
    const breakdown = document.getElementById('pfa-breakdown');
    breakdown.innerHTML = Object.entries(PFA_CATEGORIES).map(([k, v]) => {
        const time = m.totals[k] || 0;
        const pct = m.totalTime > 0 ? (time / m.totalTime) * 100 : 0;
        return `<div class="calc-breakdown-row">
            <span style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 12px; height: 12px; background: ${v.color}; border-radius: 2px;"></span>
                ${k} - ${v.name}
            </span>
            <span>${time.toFixed(1)} min (${pct.toFixed(0)}%)</span>
        </div>`;
    }).join('');

    // Baseline comparison
    if (pfaBaseline) {
        const delta = document.getElementById('pfa-baseline-delta');
        const bm = pfaBaseline.metrics;
        const timeDiff = m.totalTime - bm.totalTime;
        const distDiff = m.totalDist - bm.totalDist;
        const ratioDiff = m.processRatio - bm.processRatio;

        delta.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; text-align: center;">
                <div>
                    <div style="font-size: 11px; color: var(--text-dim);">Time</div>
                    <div style="font-size: 18px; font-weight: 600; color: ${timeDiff <= 0 ? '#4a9f6e' : '#e74c3c'};">
                        ${timeDiff <= 0 ? '' : '+'}${timeDiff.toFixed(0)} min
                    </div>
                </div>
                <div>
                    <div style="font-size: 11px; color: var(--text-dim);">Distance</div>
                    <div style="font-size: 18px; font-weight: 600; color: ${distDiff <= 0 ? '#4a9f6e' : '#e74c3c'};">
                        ${distDiff <= 0 ? '' : '+'}${distDiff.toFixed(0)} m
                    </div>
                </div>
                <div>
                    <div style="font-size: 11px; color: var(--text-dim);">Process Ratio</div>
                    <div style="font-size: 18px; font-weight: 600; color: ${ratioDiff >= 0 ? '#4a9f6e' : '#e74c3c'};">
                        ${ratioDiff >= 0 ? '+' : ''}${ratioDiff.toFixed(0)}%
                    </div>
                </div>
            </div>
        `;
        document.getElementById('pfa-baseline-compare').style.display = 'block';
    }

    // Chart
    renderPFAChart(m);
    renderPFAFlow();
}

function renderPFAChart(m) {
    const cats = Object.keys(PFA_CATEGORIES);
    const colors = cats.map(k => PFA_CATEGORIES[k].color);
    const values = cats.map(k => m.totals[k] || 0);
    const labels = cats.map(k => PFA_CATEGORIES[k].name);

    Plotly.newPlot('pfa-chart', [{
        type: 'pie',
        values: values,
        labels: labels,
        marker: { colors: colors },
        textinfo: 'label+percent',
        hole: 0.4
    }], {
        margin: { t: 20, b: 20, l: 20, r: 20 },
        paper_bgcolor: 'transparent',
        showlegend: false
    }, { responsive: true, displayModeBar: false });
}

function renderPFAFlow() {
    const container = document.getElementById('pfa-flow');
    if (!container || pfaSteps.length === 0) {
        container.innerHTML = '<div style="color: var(--text-dim); text-align: center;">Add steps to see flow</div>';
        return;
    }

    container.innerHTML = '<div style="display: flex; align-items: center; gap: 4px; flex-wrap: wrap;">' +
        pfaSteps.map((s, i) => {
            const cat = PFA_CATEGORIES[s.cat];
            return `<div style="display: flex; flex-direction: column; align-items: center; min-width: 60px;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background: ${cat.color}; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 11px;">${s.cat}</div>
                <div style="font-size: 10px; color: var(--text-dim); margin-top: 4px; text-align: center; max-width: 70px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${s.desc || '—'}</div>
                <div style="font-size: 10px; color: var(--text-secondary);">${s.time}m</div>
            </div>` + (i < pfaSteps.length - 1 ? '<div style="color: var(--text-dim);">→</div>' : '');
        }).join('') + '</div>';
}

// ============================================================================
// WFA — Workflow Analysis (Therbligs)
// ============================================================================

const WFA_CATEGORIES = {
    'VA': { name: 'Value Added', color: '#4a9f6e', type: 'VA' },
    'RW': { name: 'Required Work', color: '#3498db', type: 'NVA/R' },
    'P': { name: 'Parts', color: '#9b59b6', type: 'NVA/R' },
    'T': { name: 'Tools/Tooling', color: '#e67e22', type: 'NVA/R' },
    'I': { name: 'Inspection', color: '#f39c12', type: 'NVA/R' },
    'MH': { name: 'Material Handling', color: '#1abc9c', type: 'NVA/R' },
    'UW': { name: 'Unnecessary Work', color: '#e74c3c', type: 'NVA/N' },
    'IT': { name: 'Idle Time', color: '#95a5a6', type: 'NVA/N' }
};

let wfaElements = [];
let wfaBaseline = null;

function renderWFAElements() {
    const container = document.getElementById('wfa-elements');
    if (!container) return;

    container.innerHTML = wfaElements.map((el, i) => `
        <div style="display: grid; grid-template-columns: 40px 1fr 120px 80px 30px; gap: 8px; align-items: center; padding: 12px; background: var(--bg-secondary); border-radius: 8px;">
            <span style="font-size: 12px; color: var(--text-dim); text-align: center;">${i + 1}</span>
            <input type="text" value="${el.desc}" placeholder="Element description"
                   style="padding: 8px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary);"
                   oninput="updateWFAElement(${i}, 'desc', this.value)">
            <select style="padding: 8px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary);"
                    onchange="updateWFAElement(${i}, 'cat', this.value)">
                ${Object.entries(WFA_CATEGORIES).map(([k, v]) =>
                    `<option value="${k}" ${el.cat === k ? 'selected' : ''}>${k} - ${v.name}</option>`
                ).join('')}
            </select>
            <div style="display: flex; align-items: center; gap: 4px;">
                <input type="number" value="${el.time}" min="0" step="1"
                       style="width: 50px; padding: 6px; text-align: right; background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary);"
                       oninput="updateWFAElement(${i}, 'time', parseFloat(this.value) || 0)">
                <span style="font-size: 11px; color: var(--text-dim);">sec</span>
            </div>
            <button onclick="removeWFAElement(${i})" style="padding: 4px 8px; background: transparent; border: none; color: var(--text-dim); cursor: pointer; font-size: 16px;">&times;</button>
        </div>
    `).join('');

    calcWFA();
}

function addWFAElement() {
    wfaElements.push({ desc: '', cat: 'VA', time: 5 });
    renderWFAElements();
}

function removeWFAElement(idx) {
    wfaElements.splice(idx, 1);
    renderWFAElements();
}

function updateWFAElement(idx, field, value) {
    wfaElements[idx][field] = value;
    calcWFA();
}

function loadWFAExample() {
    wfaElements = [
        { desc: 'Reach for part', cat: 'P', time: 3 },
        { desc: 'Grasp part', cat: 'P', time: 2 },
        { desc: 'Position part', cat: 'RW', time: 4 },
        { desc: 'Secure part in fixture', cat: 'RW', time: 5 },
        { desc: 'Reach for tool', cat: 'T', time: 3 },
        { desc: 'Apply fastener', cat: 'VA', time: 8 },
        { desc: 'Apply fastener', cat: 'VA', time: 8 },
        { desc: 'Apply fastener', cat: 'VA', time: 8 },
        { desc: 'Apply fastener', cat: 'VA', time: 8 },
        { desc: 'Set down tool', cat: 'T', time: 2 },
        { desc: 'Visually inspect', cat: 'I', time: 5 },
        { desc: 'Remove from fixture', cat: 'RW', time: 4 },
        { desc: 'Place in tote', cat: 'MH', time: 3 },
        { desc: 'Wait for next part', cat: 'IT', time: 6 },
        { desc: 'Look for supervisor (question)', cat: 'UW', time: 12 },
    ];
    renderWFAElements();
}

function captureWFABaseline() {
    wfaBaseline = {
        elements: JSON.parse(JSON.stringify(wfaElements)),
        metrics: calcWFAMetrics(),
        capturedAt: new Date().toISOString()
    };
    document.getElementById('wfa-clear-baseline').style.display = 'inline-block';
    calcWFA();
}

function clearWFABaseline() {
    wfaBaseline = null;
    document.getElementById('wfa-clear-baseline').style.display = 'none';
    document.getElementById('wfa-baseline-compare').style.display = 'none';
    calcWFA();
}

function calcWFAMetrics() {
    const totals = {};
    Object.keys(WFA_CATEGORIES).forEach(k => totals[k] = 0);
    let totalTime = 0;

    wfaElements.forEach(el => {
        totals[el.cat] = (totals[el.cat] || 0) + el.time;
        totalTime += el.time;
    });

    const vaTime = totals['VA'] || 0;
    const nvarTime = (totals['RW'] || 0) + (totals['P'] || 0) + (totals['T'] || 0) + (totals['I'] || 0) + (totals['MH'] || 0);
    const nvanTime = (totals['UW'] || 0) + (totals['IT'] || 0);

    const vaRatio = totalTime > 0 ? (vaTime / totalTime) * 100 : 0;
    const nvarRatio = totalTime > 0 ? (nvarTime / totalTime) * 100 : 0;
    const nvanRatio = totalTime > 0 ? (nvanTime / totalTime) * 100 : 0;

    return { totals, totalTime, vaTime, nvarTime, nvanTime, vaRatio, nvarRatio, nvanRatio, elementCount: wfaElements.length };
}

function calcWFA() {
    const m = calcWFAMetrics();

    document.getElementById('wfa-va-ratio').innerHTML = `${m.vaRatio.toFixed(0)}<span class="calc-result-unit">%</span>`;
    document.getElementById('wfa-total-time').innerHTML = `${m.totalTime.toFixed(0)}<span class="calc-result-unit">sec</span>`;
    document.getElementById('wfa-nvar').innerHTML = `${m.nvarRatio.toFixed(0)}<span class="calc-result-unit">%</span>`;
    document.getElementById('wfa-nvan').innerHTML = `${m.nvanRatio.toFixed(0)}<span class="calc-result-unit">%</span>`;

    // Breakdown
    const breakdown = document.getElementById('wfa-breakdown');
    breakdown.innerHTML = Object.entries(WFA_CATEGORIES).map(([k, v]) => {
        const time = m.totals[k] || 0;
        const pct = m.totalTime > 0 ? (time / m.totalTime) * 100 : 0;
        return `<div class="calc-breakdown-row">
            <span style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 12px; height: 12px; background: ${v.color}; border-radius: 2px;"></span>
                ${k} - ${v.name} <span style="font-size: 10px; color: var(--text-dim);">(${v.type})</span>
            </span>
            <span>${time} sec (${pct.toFixed(0)}%)</span>
        </div>`;
    }).join('');

    // NVA/R and NVA/N lists
    const nvarList = wfaElements.filter(el => ['RW', 'P', 'T', 'I', 'MH'].includes(el.cat));
    const nvanList = wfaElements.filter(el => ['UW', 'IT'].includes(el.cat));

    document.getElementById('wfa-nvar-list').innerHTML = nvarList.length > 0
        ? nvarList.map(el => `<div style="padding: 4px 0; border-bottom: 1px solid var(--border);">${el.desc || '(no description)'} — ${el.time}s</div>`).join('')
        : '<div style="color: var(--text-dim);">No required NVA elements</div>';

    document.getElementById('wfa-nvan-list').innerHTML = nvanList.length > 0
        ? nvanList.map(el => `<div style="padding: 4px 0; border-bottom: 1px solid var(--border);">${el.desc || '(no description)'} — ${el.time}s</div>`).join('')
        : '<div style="color: var(--text-dim);">No unnecessary waste — nice!</div>';

    // Baseline comparison
    if (wfaBaseline) {
        const delta = document.getElementById('wfa-baseline-delta');
        const bm = wfaBaseline.metrics;
        const timeDiff = m.totalTime - bm.totalTime;
        const vaDiff = m.vaRatio - bm.vaRatio;
        const nvanDiff = m.nvanTime - bm.nvanTime;

        delta.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; text-align: center;">
                <div>
                    <div style="font-size: 11px; color: var(--text-dim);">Total Time</div>
                    <div style="font-size: 18px; font-weight: 600; color: ${timeDiff <= 0 ? '#4a9f6e' : '#e74c3c'};">
                        ${timeDiff <= 0 ? '' : '+'}${timeDiff} sec
                    </div>
                </div>
                <div>
                    <div style="font-size: 11px; color: var(--text-dim);">VA Ratio</div>
                    <div style="font-size: 18px; font-weight: 600; color: ${vaDiff >= 0 ? '#4a9f6e' : '#e74c3c'};">
                        ${vaDiff >= 0 ? '+' : ''}${vaDiff.toFixed(0)}%
                    </div>
                </div>
                <div>
                    <div style="font-size: 11px; color: var(--text-dim);">Waste (NVA/N)</div>
                    <div style="font-size: 18px; font-weight: 600; color: ${nvanDiff <= 0 ? '#4a9f6e' : '#e74c3c'};">
                        ${nvanDiff <= 0 ? '' : '+'}${nvanDiff} sec
                    </div>
                </div>
            </div>
        `;
        document.getElementById('wfa-baseline-compare').style.display = 'block';
    }

    // Chart
    renderWFAChart(m);
}

function renderWFAChart(m) {
    const cats = Object.keys(WFA_CATEGORIES);
    const colors = cats.map(k => WFA_CATEGORIES[k].color);
    const values = cats.map(k => m.totals[k] || 0);
    const labels = cats.map(k => WFA_CATEGORIES[k].name);

    Plotly.newPlot('wfa-chart', [{
        type: 'bar',
        x: labels,
        y: values,
        marker: { color: colors }
    }], {
        margin: { t: 20, b: 60, l: 40, r: 20 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: { tickangle: -45, color: '#a0a0a0' },
        yaxis: { title: 'Seconds', color: '#a0a0a0', gridcolor: 'rgba(160, 160, 160, 0.2)' }
    }, { responsive: true, displayModeBar: false });
}
