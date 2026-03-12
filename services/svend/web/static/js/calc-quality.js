/**
 * calc-quality.js — Quality & Statistical Calculators
 *
 * Load order: after calc-core.js, svend-math.js (uses SvendOps, renderNextSteps, MonteCarlo)
 * Extracted from: calculators.html (inline script)
 *
 * Provides: RTY (Rolled Throughput Yield), DPMO/Sigma Level, FMEA/RPN,
 * Cp/Cpk capability, sample size calculator, interactive power analysis.
 */

// ============================================================================
// RTY (Rolled Throughput Yield)
// ============================================================================

let rtyData = [
    { name: 'Step 1', fpy: 98 },
    { name: 'Step 2', fpy: 95 },
    { name: 'Step 3', fpy: 92 },
];

function renderRTYInputs() {
    const container = document.getElementById('rty-steps');
    container.innerHTML = rtyData.map((s, i) => `
        <div class="yamazumi-station">
            <span class="yamazumi-station-name">${s.name}</span>
            <input type="number" value="${s.fpy}" step="0.1" max="100" oninput="updateRTY(${i}, this.value)">
            <span style="color: var(--text-dim);">% FPY</span>
            <button class="yamazumi-station-remove" onclick="removeRTYStep(${i})">&times;</button>
        </div>
    `).join('');
    calcRTY();
}

function addRTYStep() {
    rtyData.push({ name: `Step ${rtyData.length + 1}`, fpy: 95 });
    renderRTYInputs();
}

function removeRTYStep(idx) {
    rtyData.splice(idx, 1);
    renderRTYInputs();
}

function updateRTY(idx, value) {
    rtyData[idx].fpy = parseFloat(value) || 0;
    calcRTY();
}

function calcRTY() {
    if (rtyData.length === 0) return;

    const rty = rtyData.reduce((acc, s) => acc * (s.fpy / 100), 1) * 100;
    const defects = 100 - rty;

    document.getElementById('rty-result').innerHTML = `${rty.toFixed(1)}<span class="calc-result-unit">%</span>`;
    document.getElementById('rty-defects').innerHTML = `${defects.toFixed(1)}<span class="calc-result-unit">%</span>`;
    document.getElementById('rty-hidden').textContent = `${defects.toFixed(1)}% rework/scrap hidden in process`;

    // Update derivation
    const rtySteps = rtyData.map((s, i) => {
        const cumYield = rtyData.slice(0, i + 1).reduce((acc, x) => acc * (x.fpy / 100), 1) * 100;
        return `<div class="step">
            <div class="step-num">${s.name}: FPY = ${s.fpy}%</div>
            Cumulative yield after this step: ${i === 0 ? s.fpy : `${rtyData.slice(0, i).reduce((acc, x) => acc * (x.fpy / 100), 1) * 100 > 0 ? (rtyData.slice(0, i).reduce((acc, x) => acc * (x.fpy / 100), 1) * 100).toFixed(1) : '0'}% × ${s.fpy}%`} = <strong>${cumYield.toFixed(1)}%</strong>
        </div>`;
    }).join('');
    document.getElementById('rty-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Formula</div>
            <span class="formula">RTY = FPY₁ × FPY₂ × ... × FPYₙ</span><br>
            = ${rtyData.map(s => (s.fpy / 100).toFixed(4)).join(' × ')} = <strong>${(rty / 100).toFixed(4)} = ${rty.toFixed(1)}%</strong>
        </div>
        ${rtySteps}
        <div class="step">
            <div class="step-num">Hidden Factory</div>
            ${defects.toFixed(1)}% of units require rework or become scrap — this waste is often invisible in traditional yield reporting.
        </div>
    `;

    // Waterfall chart
    const names = ['Start', ...rtyData.map(s => s.name), 'RTY'];
    const yields = [100];
    let cumulative = 100;
    rtyData.forEach(s => {
        cumulative *= s.fpy / 100;
        yields.push(cumulative);
    });

    Plotly.newPlot('rty-chart', [{
        x: names,
        y: yields,
        type: 'bar',
        marker: {
            color: yields.map((y, i) => i === yields.length - 1 ? '#4a9f6e' : '#3a7f8f')
        },
        text: yields.map(y => `${y.toFixed(1)}%`),
        textposition: 'outside'
    }], {
        margin: { t: 20, b: 60, l: 50, r: 20 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a' },
        yaxis: { title: 'Cumulative Yield (%)', range: [0, 110], gridcolor: 'rgba(255,255,255,0.1)' },
        xaxis: { gridcolor: 'rgba(255,255,255,0.1)' }
    }, { responsive: true, displayModeBar: false });

    renderNextSteps('rty-next-steps', [
        { title: 'Cycle Time Analysis', desc: 'Find where rework adds hidden cycle time', calcId: 'cycletime' },
        { title: 'Run FMEA', desc: 'Prioritize failure modes in low-yield steps', calcId: 'fmea' },
        { title: 'Check DPMO', desc: 'Convert yield to sigma level', calcId: 'dpmo' },
    ]);

    SvendOps.publish('rty', parseFloat(rty.toFixed(1)), '%', 'RTY');
}

// ============================================================================
// DPMO / Sigma Level
// ============================================================================

function updateDPMOMode() {
    const mode = document.getElementById('dpmo-mode').value;
    document.getElementById('dpmo-defects-group').style.display = mode === 'defects' ? 'flex' : 'none';
    document.getElementById('dpmo-units-group').style.display = mode === 'defects' ? 'flex' : 'none';
    document.getElementById('dpmo-opp-group').style.display = mode === 'defects' ? 'flex' : 'none';
    document.getElementById('dpmo-direct-group').style.display = mode === 'dpmo' ? 'flex' : 'none';
    document.getElementById('dpmo-yield-group').style.display = mode === 'yield' ? 'flex' : 'none';
    calcDPMO();
}

function dpmoToSigma(dpmo) {
    // Approximate conversion using normal distribution
    const yield_ = 1 - dpmo / 1000000;
    if (yield_ >= 1) return 6;
    if (yield_ <= 0) return 0;
    // Using approximation: sigma ≈ 0.8406 + sqrt(29.37 - 2.221 * ln(DPMO))
    if (dpmo <= 0) return 6;
    const sigma = 0.8406 + Math.sqrt(29.37 - 2.221 * Math.log(dpmo));
    return Math.max(0, Math.min(6, sigma));
}

function calcDPMO() {
    const mode = document.getElementById('dpmo-mode').value;
    let dpmo, yield_;

    if (mode === 'defects') {
        const defects = parseFloat(document.getElementById('dpmo-defects').value) || 0;
        const units = parseFloat(document.getElementById('dpmo-units').value) || 1;
        const opp = parseFloat(document.getElementById('dpmo-opp').value) || 1;
        dpmo = (defects / (units * opp)) * 1000000;
        yield_ = (1 - dpmo / 1000000) * 100;
    } else if (mode === 'dpmo') {
        dpmo = parseFloat(document.getElementById('dpmo-direct').value) || 0;
        yield_ = (1 - dpmo / 1000000) * 100;
    } else {
        yield_ = parseFloat(document.getElementById('dpmo-yield').value) || 0;
        dpmo = (1 - yield_ / 100) * 1000000;
    }

    const sigma = dpmoToSigma(dpmo);

    document.getElementById('dpmo-result').textContent = dpmo.toLocaleString(undefined, { maximumFractionDigits: 0 });
    document.getElementById('dpmo-sigma').innerHTML = `${sigma.toFixed(2)}<span class="calc-result-unit">σ</span>`;
    document.getElementById('dpmo-yield-result').innerHTML = `${yield_.toFixed(4)}<span class="calc-result-unit">%</span>`;
    document.getElementById('dpmo-rate').textContent = `${(100 - yield_).toFixed(4)}%`;

    // Update derivation
    let dpmoDerivation = '';
    if (mode === 'defects') {
        const defects = parseFloat(document.getElementById('dpmo-defects').value) || 0;
        const units = parseFloat(document.getElementById('dpmo-units').value) || 1;
        const opp = parseFloat(document.getElementById('dpmo-opp').value) || 1;
        dpmoDerivation = `
            <div class="step">
                <div class="step-num">Step 1: Calculate DPMO</div>
                <span class="formula">DPMO = (Defects ÷ (Units × Opportunities)) × 1,000,000</span><br>
                = (${defects} ÷ (${units} × ${opp})) × 1,000,000<br>
                = (${defects} ÷ ${units * opp}) × 1,000,000 = <strong>${Math.round(dpmo).toLocaleString()} DPMO</strong>
            </div>`;
    } else if (mode === 'dpmo') {
        dpmoDerivation = `
            <div class="step">
                <div class="step-num">Step 1: Direct DPMO Input</div>
                DPMO = <strong>${Math.round(dpmo).toLocaleString()}</strong>
            </div>`;
    } else {
        dpmoDerivation = `
            <div class="step">
                <div class="step-num">Step 1: Convert Yield to DPMO</div>
                <span class="formula">DPMO = (1 − Yield/100) × 1,000,000</span><br>
                = (1 − ${yield_.toFixed(2)}/100) × 1,000,000 = <strong>${Math.round(dpmo).toLocaleString()} DPMO</strong>
            </div>`;
    }
    dpmoDerivation += `
        <div class="step">
            <div class="step-num">Step 2: Calculate Sigma Level</div>
            <span class="formula">σ ≈ 0.8406 + √(29.37 − 2.221 × ln(DPMO))</span><br>
            Sigma Level = <strong>${sigma.toFixed(2)}σ</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 3: Process Yield</div>
            <span class="formula">Yield = (1 − DPMO/1,000,000) × 100</span><br>
            = <strong>${yield_.toFixed(4)}%</strong>
        </div>`;
    document.getElementById('dpmo-derivation-body').innerHTML = dpmoDerivation;

    // Publish to shared state
    SvendOps.publish('sigma', parseFloat(sigma.toFixed(2)), 'σ', 'DPMO/Sigma');
    SvendOps.publish('dpmo', Math.round(dpmo), 'DPMO', 'DPMO/Sigma');

    // Sigma gauge chart
    Plotly.newPlot('dpmo-chart', [{
        type: 'indicator', mode: 'gauge+number',
        value: sigma,
        number: { suffix: 'σ', font: { size: 28, color: '#e0e0e0' } },
        gauge: {
            axis: { range: [0, 6], dtick: 1, tickfont: { color: '#9aaa9a' } },
            bar: { color: sigma >= 4.5 ? '#27ae60' : sigma >= 3 ? '#f39c12' : '#e74c3c' },
            bgcolor: 'rgba(255,255,255,0.05)',
            steps: [
                { range: [0, 2], color: 'rgba(231,76,60,0.25)' },
                { range: [2, 3], color: 'rgba(243,156,18,0.2)' },
                { range: [3, 4.5], color: 'rgba(243,156,18,0.15)' },
                { range: [4.5, 6], color: 'rgba(39,174,96,0.25)' }
            ],
            threshold: { line: { color: '#27ae60', width: 3 }, value: 4.5 }
        }
    }], {
        margin: { t: 30, b: 10, l: 30, r: 30 },
        paper_bgcolor: 'transparent',
        font: { color: '#9aaa9a' },
        annotations: [
            { x: 0.25, y: -0.05, text: 'Typical: 3σ', showarrow: false, font: { size: 10, color: '#f39c12' } },
            { x: 0.75, y: -0.05, text: 'World Class: 6σ', showarrow: false, font: { size: 10, color: '#27ae60' } }
        ]
    }, { responsive: true, displayModeBar: false });

    renderNextSteps('dpmo-next-steps', [
        { title: 'RTY Analysis', desc: 'Calculate rolled throughput yield', calcId: 'rty' },
        { title: 'Cpk Check', desc: 'Evaluate process capability index', calcId: 'cpk' },
        { title: 'Cost of Quality', desc: 'Estimate cost impact of defects', calcId: 'coq' },
    ]);
}

// ============================================================================
// FMEA / RPN
// ============================================================================

let fmeaData = [
    { mode: 'Seal failure', severity: 8, occurrence: 4, detection: 5 },
    { mode: 'Wrong label', severity: 5, occurrence: 3, detection: 2 },
    { mode: 'Underfill', severity: 7, occurrence: 2, detection: 3 },
];

function renderFMEAInputs() {
    const container = document.getElementById('fmea-items');
    container.innerHTML = fmeaData.map((f, i) => `
        <div style="display: grid; grid-template-columns: 1fr repeat(3, 80px) 30px; gap: 8px; align-items: center; padding: 12px; background: var(--bg-secondary); border-radius: 8px;">
            <input type="text" value="${f.mode}" placeholder="Failure Mode" style="padding: 8px;" oninput="updateFMEA(${i}, 'mode', this.value)">
            <div>
                <label style="font-size: 10px; margin-bottom: 2px;">S</label>
                <input type="number" value="${f.severity}" min="1" max="10" style="width: 100%; text-align: center;" oninput="updateFMEA(${i}, 'severity', this.value)">
            </div>
            <div>
                <label style="font-size: 10px; margin-bottom: 2px;">O</label>
                <input type="number" value="${f.occurrence}" min="1" max="10" style="width: 100%; text-align: center;" oninput="updateFMEA(${i}, 'occurrence', this.value)">
            </div>
            <div>
                <label style="font-size: 10px; margin-bottom: 2px;">D</label>
                <input type="number" value="${f.detection}" min="1" max="10" style="width: 100%; text-align: center;" oninput="updateFMEA(${i}, 'detection', this.value)">
            </div>
            <button class="yamazumi-station-remove" onclick="removeFMEA(${i})">&times;</button>
        </div>
    `).join('');
    calcFMEA();
}

function addFMEAItem() {
    fmeaData.push({ mode: 'New failure mode', severity: 5, occurrence: 5, detection: 5 });
    renderFMEAInputs();
}

function removeFMEA(idx) {
    fmeaData.splice(idx, 1);
    renderFMEAInputs();
}

function updateFMEA(idx, field, value) {
    if (field !== 'mode') value = Math.min(10, Math.max(1, parseInt(value) || 1));
    fmeaData[idx][field] = value;
    calcFMEA();
}

function calcFMEA() {
    const rpns = fmeaData.map(f => ({ ...f, rpn: f.severity * f.occurrence * f.detection }));
    rpns.sort((a, b) => b.rpn - a.rpn);

    // Update derivation
    const fmeaSteps = rpns.map(f => `
        <div class="step">
            <div class="step-num">${f.mode}</div>
            <span class="formula">RPN = S × O × D</span> = ${f.severity} × ${f.occurrence} × ${f.detection} = <strong>${f.rpn}</strong>
            ${f.rpn > 100 ? ' <span style="color:#e74c3c;">⚠ Action required</span>' : f.rpn > 50 ? ' <span style="color:#f39c12;">Monitor</span>' : ' <span style="color:#4a9f6e;">Acceptable</span>'}
        </div>`).join('');
    document.getElementById('fmea-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Formula</div>
            <span class="formula">Risk Priority Number = Severity × Occurrence × Detection</span><br>
            Each factor: 1 (best) to 10 (worst). RPN range: 1–1,000.
        </div>
        ${fmeaSteps}
    `;

    Plotly.newPlot('fmea-chart', [{
        y: rpns.map(f => f.mode),
        x: rpns.map(f => f.rpn),
        type: 'bar',
        orientation: 'h',
        marker: { color: rpns.map(f => f.rpn > 100 ? '#e74c3c' : f.rpn > 50 ? '#f39c12' : '#4a9f6e') },
        text: rpns.map(f => `RPN: ${f.rpn}`),
        textposition: 'outside'
    }], {
        shapes: [{
            type: 'line', x0: 100, x1: 100, y0: -0.5, y1: rpns.length - 0.5,
            line: { color: '#e74c3c', width: 2, dash: 'dash' }
        }],
        margin: { t: 20, b: 40, l: 120, r: 60 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a' },
        xaxis: { title: 'Risk Priority Number (RPN)', gridcolor: 'rgba(255,255,255,0.1)' },
        yaxis: { gridcolor: 'rgba(255,255,255,0.1)' }
    }, { responsive: true, displayModeBar: false });
}

// ============================================================================
// Cp / Cpk
// ============================================================================

function calcCpk() {
    const usl = parseFloat(document.getElementById('cpk-usl').value);
    const lsl = parseFloat(document.getElementById('cpk-lsl').value);
    const mean = parseFloat(document.getElementById('cpk-mean').value);
    const std = parseFloat(document.getElementById('cpk-std').value) || 0.001;

    const cp = (usl - lsl) / (6 * std);
    const cpu = (usl - mean) / (3 * std);
    const cpl = (mean - lsl) / (3 * std);
    const cpk = Math.min(cpu, cpl);

    // Estimate defect rate from Cpk
    const z = cpk * 3;
    const defectRate = 2 * (1 - normalCDF(z)) * 100;

    document.getElementById('cpk-cp').textContent = cp.toFixed(2);
    document.getElementById('cpk-cpk').textContent = cpk.toFixed(2);
    document.getElementById('cpk-cpu').textContent = cpu.toFixed(2);
    document.getElementById('cpk-cpl').textContent = cpl.toFixed(2);
    document.getElementById('cpk-defects').textContent = defectRate < 0.01 ? '<0.01%' : `${defectRate.toFixed(2)}%`;

    // Update derivation
    document.getElementById('cpk-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Process Capability (Cp)</div>
            <span class="formula">Cp = (USL − LSL) ÷ (6σ)</span><br>
            = (${usl} − ${lsl}) ÷ (6 × ${std}) = ${(usl - lsl).toFixed(3)} ÷ ${(6 * std).toFixed(3)} = <strong>${cp.toFixed(2)}</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 2: Upper Capability Index (Cpu)</div>
            <span class="formula">Cpu = (USL − X̄) ÷ (3σ)</span><br>
            = (${usl} − ${mean}) ÷ (3 × ${std}) = ${(usl - mean).toFixed(3)} ÷ ${(3 * std).toFixed(3)} = <strong>${cpu.toFixed(2)}</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 3: Lower Capability Index (Cpl)</div>
            <span class="formula">Cpl = (X̄ − LSL) ÷ (3σ)</span><br>
            = (${mean} − ${lsl}) ÷ (3 × ${std}) = ${(mean - lsl).toFixed(3)} ÷ ${(3 * std).toFixed(3)} = <strong>${cpl.toFixed(2)}</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 4: Process Capability Index (Cpk)</div>
            <span class="formula">Cpk = min(Cpu, Cpl)</span><br>
            = min(${cpu.toFixed(2)}, ${cpl.toFixed(2)}) = <strong>${cpk.toFixed(2)}</strong>
            ${cpk >= 1.33 ? ' — Excellent' : cpk >= 1.0 ? ' — Acceptable' : cpk >= 0.67 ? ' — Poor' : ' — Incapable'}
        </div>
    `;

    let assessment = '';
    if (cpk >= 1.33) assessment = 'Excellent - process is capable';
    else if (cpk >= 1.0) assessment = 'Acceptable - monitor closely';
    else if (cpk >= 0.67) assessment = 'Poor - improvement needed';
    else assessment = 'Incapable - immediate action required';

    if (Math.abs(cpu - cpl) > 0.3) {
        assessment += (cpu < cpl) ? ', shift process down' : ', shift process up';
    }
    document.getElementById('cpk-assessment').textContent = assessment;

    // Distribution chart
    const xMin = lsl - 2 * std;
    const xMax = usl + 2 * std;
    const step = (xMax - xMin) / 100;
    const xVals = [], yVals = [];

    for (let x = xMin; x <= xMax; x += step) {
        xVals.push(x);
        yVals.push(Math.exp(-0.5 * Math.pow((x - mean) / std, 2)) / (std * Math.sqrt(2 * Math.PI)));
    }

    Plotly.newPlot('cpk-chart', [{
        x: xVals, y: yVals, type: 'scatter', mode: 'lines',
        fill: 'tozeroy', fillcolor: 'rgba(74, 159, 110, 0.3)',
        line: { color: '#4a9f6e', width: 2 }
    }], {
        shapes: [
            { type: 'line', x0: lsl, x1: lsl, y0: 0, y1: Math.max(...yVals) * 1.1, line: { color: '#e74c3c', width: 2 } },
            { type: 'line', x0: usl, x1: usl, y0: 0, y1: Math.max(...yVals) * 1.1, line: { color: '#e74c3c', width: 2 } },
            { type: 'line', x0: mean, x1: mean, y0: 0, y1: Math.max(...yVals) * 1.1, line: { color: '#3a7f8f', width: 2, dash: 'dash' } }
        ],
        annotations: [
            { x: lsl, y: Math.max(...yVals) * 1.05, text: 'LSL', showarrow: false, font: { color: '#e74c3c' } },
            { x: usl, y: Math.max(...yVals) * 1.05, text: 'USL', showarrow: false, font: { color: '#e74c3c' } },
            { x: mean, y: Math.max(...yVals) * 0.9, text: 'X̄', showarrow: false, font: { color: '#3a7f8f' } }
        ],
        margin: { t: 20, b: 40, l: 50, r: 20 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a' },
        xaxis: { gridcolor: 'rgba(255,255,255,0.1)' },
        yaxis: { showticklabels: false, gridcolor: 'rgba(255,255,255,0.1)' }
    }, { responsive: true, displayModeBar: false });

    // Publish to shared state
    SvendOps.publish('cpk', parseFloat(cpk.toFixed(2)), '', 'Cpk');
    SvendOps.publish('cp', parseFloat(cp.toFixed(2)), '', 'Cpk');

    renderNextSteps('cpk-next-steps', [
        { title: 'Sigma Level', desc: 'Convert capability to sigma score', calcId: 'dpmo' },
        { title: 'FMEA', desc: 'Assess failure risk for this process', calcId: 'fmea' },
        { title: 'Sample Size', desc: 'Plan the sample needed to confirm capability', calcId: 'samplesize' },
    ]);
}

function normalCDF(z) {
    const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    const sign = z < 0 ? -1 : 1;
    z = Math.abs(z) / Math.sqrt(2);
    const t = 1.0 / (1.0 + p * z);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-z * z);
    return 0.5 * (1.0 + sign * y);
}

function toggleCpkMonteCarlo(btn) {
    btn.classList.toggle('active');
    const results = document.getElementById('cpk-monte-results');
    results.classList.toggle('visible');
    if (btn.classList.contains('active')) { runCpkMonteCarlo(); }
}

function runCpkMonteCarlo() {
    const mean = parseFloat(document.getElementById('cpk-mean').value) || 0;
    const std = parseFloat(document.getElementById('cpk-std').value) || 0.01;
    const usl = parseFloat(document.getElementById('cpk-usl').value) || 0;
    const lsl = parseFloat(document.getElementById('cpk-lsl').value) || 0;

    const calcCpkValue = (m, s) => {
        const cpu = (usl - m) / (3 * s);
        const cpl = (m - lsl) / (3 * s);
        return Math.min(cpu, cpl);
    };

    const inputs = [
        { value: mean, cv: 0.05 },
        { value: std, cv: 0.10 }
    ];

    const sim = MonteCarlo.simulate(calcCpkValue, inputs, 2000);

    document.getElementById('cpk-monte-mean').textContent = sim.mean.toFixed(2);
    document.getElementById('cpk-monte-p5').textContent = sim.p5.toFixed(2);
    document.getElementById('cpk-monte-p95').textContent = sim.p95.toFixed(2);
    document.getElementById('cpk-monte-std').textContent = `±${sim.std.toFixed(2)}`;

    MonteCarlo.renderHistogram('cpk-monte-chart', sim, 'Cpk Distribution (2000 runs)', '');
}

// ============================================================================
// Sample Size
// ============================================================================

function updateSampleForm() {
    const type = document.getElementById('sample-type').value;
    document.getElementById('sample-std-group').style.display = type === 'mean' || type === 'comparison' ? 'flex' : 'none';
    document.getElementById('sample-prop-group').style.display = type === 'proportion' ? 'flex' : 'none';
    document.getElementById('sample-power-group').style.display = type === 'comparison' ? 'flex' : 'none';
    document.getElementById('sample-effect-group').style.display = type === 'comparison' ? 'flex' : 'none';
    document.getElementById('sample-margin-group').style.display = type !== 'comparison' ? 'flex' : 'none';
    calcSampleSize();
}

function calcSampleSize() {
    const type = document.getElementById('sample-type').value;
    const z = parseFloat(document.getElementById('sample-conf').value);
    let n, formula;

    if (type === 'mean') {
        const margin = parseFloat(document.getElementById('sample-margin').value) || 1;
        const std = parseFloat(document.getElementById('sample-std').value) || 1;
        n = Math.pow((z * std) / margin, 2);
        formula = 'n = (Zσ/E)²';
    } else if (type === 'proportion') {
        const margin = parseFloat(document.getElementById('sample-margin').value) / 100 || 0.05;
        const p = parseFloat(document.getElementById('sample-prop').value) / 100 || 0.5;
        n = Math.pow(z, 2) * p * (1 - p) / Math.pow(margin, 2);
        formula = 'n = Z²p(1-p)/E²';
    } else {
        const std = parseFloat(document.getElementById('sample-std').value) || 1;
        const zb = parseFloat(document.getElementById('sample-power').value);
        const effect = parseFloat(document.getElementById('sample-effect').value) || 1;
        n = 2 * Math.pow((z + zb) * std / effect, 2);
        formula = 'n = 2((Zα+Zβ)σ/δ)² per group';
    }

    document.getElementById('sample-result').innerHTML = `${Math.ceil(n)}<span class="calc-result-unit">${type === 'comparison' ? 'per group' : 'samples'}</span>`;
    document.getElementById('sample-formula').textContent = formula;

    // Update derivation
    let sampleDerivation = '';
    if (type === 'mean') {
        const margin = parseFloat(document.getElementById('sample-margin').value) || 1;
        const std = parseFloat(document.getElementById('sample-std').value) || 1;
        sampleDerivation = `
            <div class="step">
                <div class="step-num">Estimating a Mean</div>
                <span class="formula">n = (Zσ / E)²</span><br>
                Where Z = ${z} (confidence), σ = ${std} (std dev), E = ${margin} (margin of error)<br>
                n = (${z} × ${std} / ${margin})² = (${(z * std / margin).toFixed(2)})² = <strong>${Math.ceil(n)} samples</strong>
            </div>`;
    } else if (type === 'proportion') {
        const margin = parseFloat(document.getElementById('sample-margin').value) / 100 || 0.05;
        const p = parseFloat(document.getElementById('sample-prop').value) / 100 || 0.5;
        sampleDerivation = `
            <div class="step">
                <div class="step-num">Estimating a Proportion</div>
                <span class="formula">n = Z²p(1−p) / E²</span><br>
                Where Z = ${z}, p = ${p.toFixed(2)}, E = ${margin.toFixed(4)}<br>
                n = ${z}² × ${p.toFixed(2)} × ${(1 - p).toFixed(2)} / ${margin.toFixed(4)}² = <strong>${Math.ceil(n)} samples</strong>
            </div>`;
    } else {
        const std = parseFloat(document.getElementById('sample-std').value) || 1;
        const zb = parseFloat(document.getElementById('sample-power').value);
        const effect = parseFloat(document.getElementById('sample-effect').value) || 1;
        sampleDerivation = `
            <div class="step">
                <div class="step-num">Two-Sample Comparison</div>
                <span class="formula">n = 2((Z_α + Z_β)σ / δ)² per group</span><br>
                Where Z_α = ${z}, Z_β = ${zb}, σ = ${std}, δ = ${effect}<br>
                n = 2 × ((${z} + ${zb}) × ${std} / ${effect})² = <strong>${Math.ceil(n)} per group</strong>
            </div>`;
    }
    document.getElementById('samplesize-derivation-body').innerHTML = sampleDerivation;
}

// ============================================================================
// Power Analysis (Interactive Sample Size Calculator)
// ============================================================================

function _paNormCDF(x) {
    const a1=0.254829592, a2=-0.284496736, a3=1.421413741, a4=-1.453152027, a5=1.061405429;
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x) / Math.SQRT2;
    const t = 1.0 / (1.0 + 0.3275911 * x);
    const y = 1 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1) * t * Math.exp(-x*x);
    return 0.5 * (1 + sign * y);
}

function _paNormPPF(p) {
    const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
               1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0];
    const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
               6.680131188771972e1, -1.328068155288572e1];
    const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
               -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0];
    const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0, 3.754408661907416e0];
    const pLow = 0.02425, pHigh = 1 - pLow;
    let q, r;
    if (p < pLow) {
        q = Math.sqrt(-2 * Math.log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    } else if (p <= pHigh) {
        q = p - 0.5; r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
    } else {
        q = Math.sqrt(-2 * Math.log(1 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    }
}

function _paPower1(n, d, alpha, sided) {
    const za = sided === 1 ? _paNormPPF(1 - alpha) : _paNormPPF(1 - alpha/2);
    return 1 - _paNormCDF(za - d * Math.sqrt(n));
}
function _paPower2(n1, d, alpha, ratio) {
    const za = _paNormPPF(1 - alpha/2);
    const n2 = Math.ceil(n1 * ratio);
    return 1 - _paNormCDF(za - d * Math.sqrt(n1 * n2 / (n1 + n2)));
}
function _paPowerProp(n, p1, p2, alpha) {
    const za = _paNormPPF(1 - alpha/2);
    const diff = Math.abs(p2 - p1);
    const pbar = (p1 + p2) / 2;
    const se0 = Math.sqrt(2 * pbar * (1 - pbar) / n);
    const se1 = Math.sqrt((p1*(1-p1) + p2*(1-p2)) / n);
    return _paNormCDF((diff - za * se0) / se1);
}
function _paPowerAnova(nPG, f, k, alpha) {
    const za = _paNormPPF(1 - alpha);
    const lambda = nPG * k * f * f;
    const df1 = k - 1;
    return _paNormCDF(Math.sqrt(2 * lambda / df1) - Math.sqrt(2 * df1 - 1) - za);
}
function _paFindN(powerFn, target) {
    for (let n = 2; n <= 10000; n++) { if (powerFn(n) >= target) return n; }
    return 10000;
}

function paUpdateSlider(sliderId, valId, dec) {
    const v = parseFloat(document.getElementById(sliderId).value);
    const el = document.getElementById(valId);
    if (dec === -1) el.textContent = Math.round(v * 100) + '%';
    else if (dec === 0) el.textContent = Math.round(v);
    else el.textContent = v.toFixed(dec || 2);
}

function paSetEffect(val) {
    const slider = document.getElementById('pa-effect');
    slider.value = val;
    paUpdateSlider('pa-effect', 'pa-effect-val');
    paUpdateBenchmark();
    paCalculate();
}

function paUpdateBenchmark() {
    const v = parseFloat(document.getElementById('pa-effect').value);
    const type = document.getElementById('pa-test-type').value;
    const benchmarks = type === 'anova'
        ? [0.10, 0.25, 0.40]
        : [0.2, 0.5, 0.8];
    document.querySelectorAll('.pa-bench').forEach((btn, i) => {
        btn.classList.toggle('active', Math.abs(benchmarks[i] - v) < 0.015);
    });
}

function paUpdateForm() {
    const type = document.getElementById('pa-test-type').value;
    const effectGroup = document.getElementById('pa-effect-group');
    const propInputs = document.getElementById('pa-prop-inputs');
    const groupsGroup = document.getElementById('pa-groups-group');
    const ratioGroup = document.getElementById('pa-ratio-group');
    const sidedGroup = document.getElementById('pa-sided-group');
    const benchmarks = document.getElementById('pa-benchmarks');
    const effectType = document.getElementById('pa-effect-type');
    const slider = document.getElementById('pa-effect');

    // Reset visibility
    effectGroup.style.display = '';
    propInputs.style.display = 'none';
    groupsGroup.style.display = 'none';
    ratioGroup.style.display = 'none';
    sidedGroup.style.display = 'none';

    if (type === 'proportion') {
        effectGroup.style.display = 'none';
        benchmarks.style.display = 'none';
        propInputs.style.display = '';
    } else if (type === 'anova') {
        effectType.textContent = "Cohen's f";
        slider.min = 0.05; slider.max = 1.0; slider.value = 0.25;
        paUpdateSlider('pa-effect', 'pa-effect-val');
        groupsGroup.style.display = '';
        benchmarks.style.display = 'flex';
        benchmarks.innerHTML =
            '<button class="pa-bench" onclick="paSetEffect(0.10)">Small (0.10)</button>' +
            '<button class="pa-bench active" onclick="paSetEffect(0.25)">Medium (0.25)</button>' +
            '<button class="pa-bench" onclick="paSetEffect(0.40)">Large (0.40)</button>';
    } else {
        effectType.textContent = "Cohen's d";
        slider.min = 0.05; slider.max = 2.0; slider.value = 0.50;
        paUpdateSlider('pa-effect', 'pa-effect-val');
        benchmarks.style.display = 'flex';
        benchmarks.innerHTML =
            '<button class="pa-bench" onclick="paSetEffect(0.2)">Small (0.2)</button>' +
            '<button class="pa-bench active" onclick="paSetEffect(0.5)">Medium (0.5)</button>' +
            '<button class="pa-bench" onclick="paSetEffect(0.8)">Large (0.8)</button>';
        if (type === 'twosample') ratioGroup.style.display = '';
        if (type === 'onesample') sidedGroup.style.display = '';
    }
}

let _paRAF = null;
function paCalculate() {
    if (_paRAF) cancelAnimationFrame(_paRAF);
    _paRAF = requestAnimationFrame(_paDoCalc);
}

function _paDoCalc() {
    const type = document.getElementById('pa-test-type').value;
    const alpha = parseFloat(document.getElementById('pa-alpha').value);
    const power = parseFloat(document.getElementById('pa-power').value);
    let n, powerFn, extraInfo = '', unit = 'per group';

    if (type === 'onesample') {
        const d = parseFloat(document.getElementById('pa-effect').value);
        const sided = parseInt(document.getElementById('pa-sided').value);
        powerFn = nn => _paPower1(nn, d, alpha, sided);
        n = _paFindN(powerFn, power);
        extraInfo = `1-sample t-test (${sided === 1 ? 'one' : 'two'}-sided), d = ${d.toFixed(2)}`;
        unit = 'samples';
    } else if (type === 'twosample') {
        const d = parseFloat(document.getElementById('pa-effect').value);
        const ratio = parseFloat(document.getElementById('pa-ratio').value);
        powerFn = nn => _paPower2(nn, d, alpha, ratio);
        n = _paFindN(powerFn, power);
        const n2 = Math.ceil(n * ratio);
        extraInfo = `2-sample t-test, d = ${d.toFixed(2)}. Group 1: ${n}, Group 2: ${n2}. Total: ${n + n2}`;
    } else if (type === 'proportion') {
        const p1 = parseFloat(document.getElementById('pa-p1').value);
        const p2 = parseFloat(document.getElementById('pa-p2').value);
        if (Math.abs(p1 - p2) < 0.001) {
            document.getElementById('pa-n-result').innerHTML = '--<span class="calc-result-unit">p1 ≠ p2</span>';
            document.getElementById('pa-achieved-result').innerHTML = '--<span class="calc-result-unit">%</span>';
            document.getElementById('pa-interpretation').innerHTML = '<span style="color:var(--warning)">p1 and p2 must differ.</span>';
            return;
        }
        powerFn = nn => _paPowerProp(nn, p1, p2, alpha);
        n = _paFindN(powerFn, power);
        extraInfo = `Two-proportion test, |p2 − p1| = ${Math.abs(p2-p1).toFixed(2)}. n per group = ${n}. Total: ${2*n}`;
    } else {
        const f = parseFloat(document.getElementById('pa-effect').value);
        const k = parseInt(document.getElementById('pa-groups').value);
        powerFn = nn => _paPowerAnova(nn, f, k, alpha);
        n = _paFindN(powerFn, power);
        extraInfo = `ANOVA with ${k} groups, f = ${f.toFixed(2)}. n per group = ${n}. Total: ${n * k}`;
    }

    const achieved = powerFn(n);
    const aPct = (achieved * 100).toFixed(1);
    const color = achieved >= 0.9 ? '#27ae60' : achieved >= 0.8 ? '#4a9f6e' : '#e8c547';

    document.getElementById('pa-n-result').innerHTML = `${n}<span class="calc-result-unit">${unit}</span>`;
    document.getElementById('pa-achieved-result').innerHTML = `${aPct}<span class="calc-result-unit">%</span>`;
    document.getElementById('pa-achieved-result').style.color = color;
    document.getElementById('pa-interpretation').innerHTML =
        `<strong>${extraInfo}.</strong> You need <strong>${n} observations ${unit}</strong> to achieve ${Math.round(power*100)}% power at α = ${alpha.toFixed(3)}. Achieved power: <strong>${aPct}%</strong>.`;

    _paDrawCurve(powerFn, n, power);
    _paBuildTable(powerFn, power);
}

function _paDrawCurve(powerFn, nOpt, targetPower) {
    const chartEl = document.getElementById('pa-power-chart');
    if (!chartEl) return;

    const maxN = Math.max(nOpt * 3, 20);
    const nSteps = Math.min(maxN, 200);
    const xs = [], ys = [];
    for (let i = 0; i <= nSteps; i++) {
        const nn = Math.max(2, Math.round(i * maxN / nSteps));
        xs.push(nn);
        ys.push(Math.min(powerFn(nn), 1));
    }

    const traces = [
        {
            x: xs, y: ys, type: 'scatter', mode: 'lines',
            name: 'Power', line: { color: '#4a9f6e', width: 2.5 },
            fill: 'tozeroy', fillcolor: 'rgba(74,159,110,0.08)',
            hovertemplate: 'n = %{x}<br>Power = %{y:.1%}<extra></extra>',
        },
        {
            x: [nOpt], y: [powerFn(nOpt)], type: 'scatter', mode: 'markers',
            name: 'Required n',
            marker: { color: '#e8c547', size: 12, line: { color: 'rgba(232,197,71,0.4)', width: 3 } },
            hovertemplate: 'n = %{x}<br>Power = %{y:.1%}<extra></extra>',
        },
    ];

    const layout = {
        margin: { t: 10, r: 20, b: 45, l: 55 },
        xaxis: { title: 'Sample Size (n)', gridcolor: 'rgba(255,255,255,0.07)', rangemode: 'tozero' },
        yaxis: { title: 'Power', gridcolor: 'rgba(255,255,255,0.07)', range: [0, 1.05],
                 tickformat: '.0%', dtick: 0.2 },
        showlegend: false,
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a', size: 11 },
        shapes: [
            { type: 'line', x0: 0, x1: maxN, y0: targetPower, y1: targetPower,
              line: { color: 'rgba(232,197,71,0.35)', width: 1.5, dash: 'dash' } },
            { type: 'line', x0: nOpt, x1: nOpt, y0: 0, y1: 1,
              line: { color: 'rgba(232,197,71,0.2)', width: 1, dash: 'dot' } },
        ],
        annotations: [
            { x: nOpt, y: powerFn(nOpt), text: `n = ${nOpt}`, showarrow: true,
              arrowhead: 0, ax: 30, ay: -30, font: { color: '#e8c547', size: 12 } },
        ],
    };

    Plotly.react(chartEl, traces, layout, { responsive: true, displayModeBar: false });
}

function _paBuildTable(powerFn, currentTarget) {
    const powers = [0.70, 0.80, 0.85, 0.90, 0.95, 0.99];
    let html = '<thead><tr><th>Target Power</th><th>Required n (per group)</th><th>Achieved Power</th></tr></thead><tbody>';
    powers.forEach(p => {
        const nn = _paFindN(powerFn, p);
        const achieved = powerFn(nn);
        const isActive = Math.abs(p - currentTarget) < 0.005;
        html += `<tr class="${isActive ? 'pa-highlight' : ''}"><td>${(p*100).toFixed(0)}%</td><td>${nn}</td><td>${(achieved*100).toFixed(1)}%</td></tr>`;
    });
    html += '</tbody>';
    document.getElementById('pa-size-table').innerHTML = html;
}
