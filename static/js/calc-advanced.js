/**
 * calc-advanced.js — Advanced Statistical Calculators
 *
 * Load order: after calc-core.js, svend-math.js
 * Extracted from: calculators.html (inline script)
 *
 * Contains: Multi-Response Desirability Optimizer, SPC Rare Events Lab (G/T Charts),
 *           Probit/Dose-Response Explorer, MTBF/MTTR + Availability,
 *           Erlang C Staffing, Risk Matrix (5x5)
 */

// ============================================================================
// MULTI-RESPONSE DESIRABILITY OPTIMIZER
// ============================================================================

const desirState = {
    responses: [],  // {name, goal, lower, target, upper, weight, importance, coeffs:[b0,b1,b2,...]}
    factors: [],    // {name, low, high}
};

function desirAddResponse(name='', goal='maximize', lower=0, target=100, upper=100, weight=1, importance=1) {
    const idx = desirState.responses.length;
    desirState.responses.push({name: name || `Y${idx+1}`, goal, lower, target, upper, weight, importance, coeffs:[]});
    desirRenderResponses();
    desirRenderModels();
}

function desirRemoveResponse(idx) {
    desirState.responses.splice(idx, 1);
    desirRenderResponses();
    desirRenderModels();
}

function desirAddFactor(name='', low=0, high=100) {
    const idx = desirState.factors.length;
    desirState.factors.push({name: name || `X${idx+1}`, low, high});
    desirRenderFactors();
    desirRenderModels();
}

function desirRemoveFactor(idx) {
    desirState.factors.splice(idx, 1);
    desirRenderFactors();
    desirRenderModels();
}

function desirRenderResponses() {
    const el = document.getElementById('desir-responses');
    if (!el) return;
    el.innerHTML = desirState.responses.map((r, i) => `
        <div style="display:grid;grid-template-columns:120px 110px 80px 80px 80px 1fr 1fr 30px;gap:8px;align-items:end;margin-bottom:8px;font-size:13px;">
            <div class="calc-input"><label>Name</label><input type="text" value="${r.name}" onchange="desirState.responses[${i}].name=this.value" style="padding:6px 8px;"></div>
            <div class="calc-input"><label>Goal</label><select onchange="desirState.responses[${i}].goal=this.value;desirGoalChanged(${i})">
                <option value="maximize" ${r.goal==='maximize'?'selected':''}>Maximize</option>
                <option value="minimize" ${r.goal==='minimize'?'selected':''}>Minimize</option>
                <option value="target" ${r.goal==='target'?'selected':''}>Target</option>
            </select></div>
            <div class="calc-input"><label>Lower</label><input type="number" value="${r.lower}" onchange="desirState.responses[${i}].lower=+this.value" style="padding:6px 8px;"></div>
            <div class="calc-input"><label>${r.goal==='target'?'Target':'Target'}</label><input type="number" value="${r.target}" onchange="desirState.responses[${i}].target=+this.value" style="padding:6px 8px;"></div>
            <div class="calc-input"><label>Upper</label><input type="number" value="${r.upper}" onchange="desirState.responses[${i}].upper=+this.value" style="padding:6px 8px;"></div>
            <div class="calc-input"><label>Weight</label><div class="slider-row"><input type="range" min="0.1" max="10" step="0.1" value="${r.weight}" oninput="desirState.responses[${i}].weight=+this.value;this.closest('.slider-row').querySelector('.slider-val').textContent=parseFloat(this.value).toFixed(1)"><span class="slider-val">${r.weight.toFixed(1)}</span></div></div>
            <div class="calc-input"><label>Importance</label><div class="slider-row"><input type="range" min="1" max="10" step="1" value="${r.importance}" oninput="desirState.responses[${i}].importance=+this.value;this.closest('.slider-row').querySelector('.slider-val').textContent=this.value"><span class="slider-val">${r.importance}</span></div></div>
            <button onclick="desirRemoveResponse(${i})" style="padding:4px 8px;background:none;border:1px solid var(--border);border-radius:4px;color:var(--text-dim);cursor:pointer;font-size:16px;margin-bottom:2px;">&times;</button>
        </div>
    `).join('');
}

function desirGoalChanged(idx) { desirRenderResponses(); }

function desirRenderFactors() {
    const el = document.getElementById('desir-factors');
    if (!el) return;
    el.innerHTML = desirState.factors.map((f, i) => `
        <div style="display:grid;grid-template-columns:140px 100px 100px 30px;gap:8px;align-items:end;margin-bottom:8px;font-size:13px;">
            <div class="calc-input"><label>Factor Name</label><input type="text" value="${f.name}" onchange="desirState.factors[${i}].name=this.value" style="padding:6px 8px;"></div>
            <div class="calc-input"><label>Low</label><input type="number" value="${f.low}" onchange="desirState.factors[${i}].low=+this.value" style="padding:6px 8px;"></div>
            <div class="calc-input"><label>High</label><input type="number" value="${f.high}" onchange="desirState.factors[${i}].high=+this.value" style="padding:6px 8px;"></div>
            <button onclick="desirRemoveFactor(${i})" style="padding:4px 8px;background:none;border:1px solid var(--border);border-radius:4px;color:var(--text-dim);cursor:pointer;font-size:16px;margin-bottom:2px;">&times;</button>
        </div>
    `).join('');
}

function desirRenderModels() {
    const el = document.getElementById('desir-models');
    if (!el) return;
    if (desirState.responses.length === 0 || desirState.factors.length === 0) {
        el.innerHTML = '<div style="color:var(--text-dim);font-size:13px;">Add responses and factors first.</div>';
        return;
    }
    el.innerHTML = desirState.responses.map((r, ri) => {
        // Ensure coeffs array is right size: b0 + one per factor
        while (r.coeffs.length < desirState.factors.length + 1) r.coeffs.push(0);
        r.coeffs.length = desirState.factors.length + 1;
        const coeffInputs = [`<span style="font-size:12px;color:var(--text-dim);">b0=</span><input type="number" value="${r.coeffs[0]}" onchange="desirState.responses[${ri}].coeffs[0]=+this.value" style="width:60px;padding:4px 6px;font-size:12px;">`];
        desirState.factors.forEach((f, fi) => {
            coeffInputs.push(`<span style="font-size:12px;color:var(--text-dim);">+ b${fi+1}(${f.name})=</span><input type="number" value="${r.coeffs[fi+1]}" onchange="desirState.responses[${ri}].coeffs[${fi+1}]=+this.value" style="width:60px;padding:4px 6px;font-size:12px;">`);
        });
        return `<div style="background:var(--bg-secondary);padding:10px 14px;border-radius:8px;"><strong style="font-size:13px;">${r.name}:</strong> <span style="display:inline-flex;align-items:center;gap:4px;flex-wrap:wrap;">${coeffInputs.join(' ')}</span></div>`;
    }).join('');
}

function desirCalcD(value, r) {
    const {goal, lower, target, upper, weight} = r;
    if (goal === 'maximize') {
        if (value <= lower) return 0;
        if (value >= target) return 1;
        return Math.pow((value - lower) / (target - lower), weight);
    } else if (goal === 'minimize') {
        if (value >= upper) return 0;
        if (value <= target) return 1;
        return Math.pow((upper - value) / (upper - target), weight);
    } else {
        if (value < lower || value > upper) return 0;
        if (value <= target) return Math.pow((value - lower) / (target - lower), weight);
        return Math.pow((upper - value) / (upper - target), weight);
    }
}

function desirPredictResponse(r, factorValues) {
    let y = r.coeffs[0] || 0;
    for (let fi = 0; fi < factorValues.length; fi++) {
        y += (r.coeffs[fi + 1] || 0) * factorValues[fi];
    }
    return y;
}

function runDesirability() {
    const responses = desirState.responses;
    const factors = desirState.factors;
    if (responses.length < 2) { alert('Add at least 2 responses.'); return; }
    if (factors.length < 1) { alert('Add at least 1 factor.'); return; }

    const gridN = factors.length <= 2 ? 31 : (factors.length === 3 ? 15 : 11);
    const totalImportance = responses.reduce((s, r) => s + r.importance, 0);

    // Grid search
    let bestD = -1, bestSettings = null, bestPredictions = null, bestIndividual = null;
    const gridSteps = factors.map(f => {
        const steps = [];
        for (let i = 0; i <= gridN; i++) steps.push(f.low + (f.high - f.low) * i / gridN);
        return steps;
    });

    // For contour: store D values for first two factors
    const contourX = gridSteps[0] || [];
    const contourY = gridSteps[1] || [];
    const contourZ = factors.length >= 2 ? contourY.map(() => contourX.map(() => 0)) : null;

    function evalPoint(fv) {
        let compositeD = 1;
        const predictions = [], individuals = [];
        for (const r of responses) {
            const yHat = desirPredictResponse(r, fv);
            const di = desirCalcD(yHat, r);
            predictions.push(yHat);
            individuals.push(di);
            compositeD *= Math.pow(di, r.importance / totalImportance);
        }
        return {compositeD, predictions, individuals};
    }

    // Recursive grid enumeration
    function enumerate(depth, fv) {
        if (depth === factors.length) {
            const {compositeD, predictions, individuals} = evalPoint(fv);
            if (compositeD > bestD) {
                bestD = compositeD;
                bestSettings = [...fv];
                bestPredictions = predictions;
                bestIndividual = individuals;
            }
            // Store contour for first 2 factors (hold others at midpoint)
            if (contourZ && depth >= 2) {
                const xi = gridSteps[0].indexOf(fv[0]);
                const yi = gridSteps[1].indexOf(fv[1]);
                if (xi >= 0 && yi >= 0) {
                    let allMid = true;
                    for (let k = 2; k < factors.length; k++) {
                        const mid = (factors[k].low + factors[k].high) / 2;
                        if (Math.abs(fv[k] - mid) > 0.001) { allMid = false; break; }
                    }
                    if (allMid) contourZ[yi][xi] = compositeD;
                }
            }
            return;
        }
        const steps = gridSteps[depth];
        // For factors beyond the first 2 in contour mode, only use midpoint for contour but search all for optimum
        for (const v of steps) {
            fv[depth] = v;
            enumerate(depth + 1, fv);
        }
    }

    // For contour: if >2 factors, run a separate 2D sweep with others at midpoint
    if (factors.length > 2) {
        const midFV = factors.map(f => (f.low + f.high) / 2);
        for (let yi = 0; yi < contourY.length; yi++) {
            for (let xi = 0; xi < contourX.length; xi++) {
                const fv = [...midFV];
                fv[0] = contourX[xi];
                fv[1] = contourY[yi];
                const {compositeD} = evalPoint(fv);
                contourZ[yi][xi] = compositeD;
            }
        }
    }

    enumerate(0, new Array(factors.length).fill(0));

    // Render desirability profiles
    const profilesEl = document.getElementById('desir-profiles');
    profilesEl.innerHTML = '';
    responses.forEach((r, i) => {
        const div = document.createElement('div');
        div.id = `desir-profile-${i}`;
        div.style.height = '220px';
        profilesEl.appendChild(div);

        const yRange = r.upper - r.lower;
        const yMin = r.lower - yRange * 0.1;
        const yMax = r.upper + yRange * 0.1;
        const nPts = 100;
        const yVals = [], dVals = [];
        for (let j = 0; j <= nPts; j++) {
            const y = yMin + (yMax - yMin) * j / nPts;
            yVals.push(y);
            dVals.push(desirCalcD(y, r));
        }

        const profileSpec = {
            title: `${r.name} (${r.goal}) — d = ${bestIndividual ? bestIndividual[i].toFixed(3) : '?'}`,
            chart_type: 'line',
            traces: [
                { x: yVals, y: dVals, name: 'd(y)', trace_type: 'line', color: '#4a9f6e', width: 2.5 },
            ],
            x_axis: { label: r.name }, y_axis: { label: 'd(y)' },
            reference_lines: [], markers: [], zones: [],
        };
        if (bestPredictions) {
            profileSpec.traces.push({ x: [bestPredictions[i]], y: [bestIndividual[i]], name: `Optimal: ${bestPredictions[i].toFixed(2)}`, trace_type: 'scatter', color: '#e74c3c', marker_size: 10 });
        }
        ForgeViz.render(document.getElementById(div.id), profileSpec);
    });

    // Render contour
    if (contourZ && factors.length >= 2) {
        const contourSpec = {
            type: 'heatmap',
            z: contourZ, x: contourX, y: contourY,
            colorscale: [[0,'#1a1a2e'],[0.5,'#f39c12'],[1,'#27ae60']],
            title: '', chart_type: 'scatter',
            traces: [],
            x_axis: { label: factors[0].name }, y_axis: { label: factors[1].name },
            reference_lines: [], markers: [], zones: [],
        };
        if (bestSettings) {
            contourSpec.markers.push({ x: bestSettings[0], y: bestSettings[1], label: 'Optimal', color: '#e74c3c' });
        }
        ForgeViz.render(document.getElementById('desir-contour'), contourSpec);
    } else if (factors.length === 1) {
        // 1D: line plot of composite D vs factor
        const xVals = contourX, dLine = [];
        for (const x of xVals) {
            const {compositeD} = evalPoint([x]);
            dLine.push(compositeD);
        }
        const contour1DSpec = {
            title: '', chart_type: 'line',
            traces: [
                { x: xVals, y: dLine, name: 'Composite D', trace_type: 'line', color: '#4a9f6e', width: 2.5 },
            ],
            x_axis: { label: factors[0].name }, y_axis: { label: 'Composite D' },
            reference_lines: [], markers: [], zones: [],
        };
        if (bestSettings) {
            contour1DSpec.traces.push({ x: [bestSettings[0]], y: [bestD], name: 'Optimal', trace_type: 'scatter', color: '#e74c3c', marker_size: 12 });
        }
        ForgeViz.render(document.getElementById('desir-contour'), contour1DSpec);
    }

    // Results cards
    const resultsEl = document.getElementById('desir-results');
    let cardsHtml = `<div class="calc-result-card primary"><div class="calc-result-label">Composite D</div><div class="calc-result-value" style="color:${bestD > 0.8 ? '#27ae60' : bestD > 0.5 ? '#f39c12' : '#e74c3c'}">${bestD.toFixed(4)}</div></div>`;
    if (bestSettings) {
        factors.forEach((f, i) => {
            cardsHtml += `<div class="calc-result-card"><div class="calc-result-label">Optimal ${f.name}</div><div class="calc-result-value">${bestSettings[i].toFixed(2)}</div></div>`;
        });
        responses.forEach((r, i) => {
            cardsHtml += `<div class="calc-result-card"><div class="calc-result-label">${r.name} (predicted)</div><div class="calc-result-value">${bestPredictions[i].toFixed(2)}</div></div>`;
        });
    }
    resultsEl.innerHTML = cardsHtml;
    document.getElementById('desir-D').textContent = bestD.toFixed(4);

    // Insight panel
    const insightEl = document.getElementById('desir-insights');
    const insightSection = document.getElementById('desir-insight-section');
    insightSection.style.display = 'block';

    // Sensitivity: perturb each factor ±5% and measure D change
    let analysisHtml = '<div><strong style="color:var(--text-primary);">Sensitivity Analysis</strong><br>';
    const sensitivities = [];
    if (bestSettings) {
        factors.forEach((f, i) => {
            const delta = (f.high - f.low) * 0.05;
            const up = [...bestSettings]; up[i] += delta;
            const dn = [...bestSettings]; dn[i] -= delta;
            const dUp = evalPoint(up).compositeD;
            const dDn = evalPoint(dn).compositeD;
            const sens = Math.abs(dUp - dDn) / (2 * delta);
            sensitivities.push({name: f.name, sens, dUp, dDn});
            const color = sens > 0.005 ? '#e74c3c' : sens > 0.001 ? '#f39c12' : '#27ae60';
            analysisHtml += `<div style="margin:4px 0;"><span style="color:${color};">&#9679;</span> <strong>${f.name}:</strong> &Delta;D/&Delta;X = ${sens.toFixed(5)} ${sens > 0.005 ? '(sensitive)' : '(robust)'}</div>`;
        });
    }
    // Binding responses
    analysisHtml += '<br><strong style="color:var(--text-primary);">Response Status</strong><br>';
    responses.forEach((r, i) => {
        const di = bestIndividual ? bestIndividual[i] : 0;
        const status = di < 0.3 ? '&#x274C; Constraining' : di < 0.7 ? '&#x26A0; Moderate' : '&#x2705; Satisfied';
        analysisHtml += `<div style="margin:4px 0;">${status} <strong>${r.name}:</strong> d = ${di.toFixed(3)}</div>`;
    });
    analysisHtml += '</div>';

    let improvHtml = '<div><strong style="color:var(--text-primary);">Improvement Suggestions</strong><br>';
    responses.forEach((r, i) => {
        const di = bestIndividual ? bestIndividual[i] : 0;
        if (di < 0.5) {
            if (r.goal === 'maximize') improvHtml += `<div style="margin:4px 0;color:#f39c12;">&#x1F4A1; ${r.name} (d=${di.toFixed(2)}): Consider raising the lower bound from ${r.lower} — currently very restrictive.</div>`;
            else if (r.goal === 'minimize') improvHtml += `<div style="margin:4px 0;color:#f39c12;">&#x1F4A1; ${r.name} (d=${di.toFixed(2)}): Consider relaxing the upper bound from ${r.upper}.</div>`;
            else improvHtml += `<div style="margin:4px 0;color:#f39c12;">&#x1F4A1; ${r.name} (d=${di.toFixed(2)}): Widen the acceptable range [${r.lower}, ${r.upper}].</div>`;
        }
    });
    if (sensitivities.length > 0) {
        const mostSens = sensitivities.sort((a,b) => b.sens - a.sens)[0];
        improvHtml += `<div style="margin:8px 0;">&#x1F50D; <strong>${mostSens.name}</strong> is the most influential factor — a confirmation experiment varying this factor would be most informative.</div>`;
    }
    if (bestD >= 0.8) improvHtml += '<div style="margin:8px 0;color:#27ae60;">&#x2705; Composite D &ge; 0.8 — optimization is in a good region. Verify with confirmation runs.</div>';
    else if (bestD < 0.3) improvHtml += '<div style="margin:8px 0;color:#e74c3c;">&#x26A0; Composite D &lt; 0.3 — responses may be fundamentally conflicting. Consider relaxing constraints or adjusting factor ranges.</div>';
    improvHtml += '</div>';
    insightEl.innerHTML = analysisHtml + improvHtml;
}

function desirLoadExample() {
    desirState.responses = [];
    desirState.factors = [];

    desirState.factors.push({name: 'Temperature', low: 60, high: 100});
    desirState.factors.push({name: 'Pressure', low: 1, high: 5});

    // Yield: maximize 60-95, Y = 30 + 0.5*T + 5*P
    desirState.responses.push({name: 'Yield %', goal: 'maximize', lower: 60, target: 95, upper: 95, weight: 1, importance: 3, coeffs: [30, 0.5, 5]});
    // Purity: target 99.5, range 98-100, Y = 102 - 0.03*T + 0.1*P
    desirState.responses.push({name: 'Purity %', goal: 'target', lower: 98, target: 99.5, upper: 100, weight: 1, importance: 5, coeffs: [102, -0.03, 0.1]});
    // Cost: minimize $10-$50, Y = -20 + 0.4*T + 3*P
    desirState.responses.push({name: 'Cost $', goal: 'minimize', lower: 10, target: 10, upper: 50, weight: 1, importance: 2, coeffs: [-20, 0.4, 3]});

    desirRenderResponses();
    desirRenderFactors();
    desirRenderModels();
    runDesirability();
}

// ============================================================================
// SPC RARE EVENTS LAB (G Chart + T Chart)
// ============================================================================

const reState = {
    data: [],
    running: false,
    interval: null,
    currentIdx: 0,
};

function reUpdateConfig() {
    const type = document.getElementById('re-type').value;
    const rateLabel = type === 'G' ? 'probability per opportunity' : 'events per time unit';
    document.querySelector('#re-rate + .input-hint').textContent = rateLabel;
}

function reGenerateData() {
    const type = document.getElementById('re-type').value;
    const rate = parseFloat(document.getElementById('re-rate').value);
    const n = parseInt(document.getElementById('re-n').value);
    const shiftAt = parseInt(document.getElementById('re-shift-at').value);
    const shiftMag = parseFloat(document.getElementById('re-shift-mag').value);

    const data = [];
    for (let i = 0; i < n; i++) {
        const currentRate = (shiftAt > 0 && i >= shiftAt) ? rate * shiftMag : rate;
        if (type === 'G') {
            // Geometric: count of opportunities between events
            // E[G] = (1-p)/p
            const p = Math.min(currentRate, 0.99);
            data.push(Math.floor(Math.log(Math.random()) / Math.log(1 - p)));
        } else {
            // Exponential: time between events
            // E[T] = 1/lambda
            data.push(-Math.log(Math.random()) / currentRate);
        }
    }
    return data;
}

function reCalcLimits(data, type) {
    const n = data.length;
    const mean = data.reduce((a, b) => a + b, 0) / n;

    if (type === 'G') {
        // G chart: based on geometric distribution
        // CL = mean, UCL/LCL from 3-sigma on geometric
        const pHat = 1 / (mean + 1);  // estimated p from mean = (1-p)/p
        const sigma = Math.sqrt((1 - pHat) / (pHat * pHat));
        return {
            cl: mean,
            ucl: mean + 3 * sigma,
            lcl: Math.max(0, mean - 3 * sigma),
            pHat, sigma, mean
        };
    } else {
        // T chart: transform T^(1/3.6) to approximate normality, then I-MR limits
        // Or simpler: use raw data with exponential-based limits
        const lambdaHat = 1 / mean;
        // For exponential: Var(T) = 1/lambda^2, so sigma = 1/lambda = mean
        return {
            cl: mean,
            ucl: mean + 3 * mean,  // = 4*mean (conservative)
            lcl: Math.max(0, mean - 3 * mean),  // 0 in practice
            lambdaHat, sigma: mean, mean
        };
    }
}

function reRenderChart(data, limits, type, highlightIdx) {
    const ooc = [];
    data.forEach((v, i) => {
        if (v > limits.ucl || v < limits.lcl) ooc.push(i);
    });

    const xIndices = data.map((_, i) => i);
    const reChartSpec = {
        title: '', chart_type: 'line',
        traces: [
            { x: xIndices, y: data, name: type === 'G' ? 'Count Between Events' : 'Time Between Events', trace_type: 'line', color: '#4a9f6e', width: 1 },
        ],
        reference_lines: [
            { value: limits.cl, axis: 'y', color: '#00b894', dash: 'dashed', label: 'CL' },
            { value: limits.ucl, axis: 'y', color: '#d63031', dash: 'dashed', label: 'UCL' },
            { value: limits.lcl, axis: 'y', color: '#d63031', dash: 'dashed', label: 'LCL' },
        ],
        markers: [],
        zones: [],
        x_axis: { label: 'Sample #' }, y_axis: { label: type === 'G' ? 'Count' : 'Time' },
    };
    if (ooc.length > 0) {
        reChartSpec.traces.push({ x: ooc, y: ooc.map(i => data[i]), name: 'OOC', trace_type: 'scatter', color: '#e74c3c', marker_size: 10 });
    }
    const shiftAt = parseInt(document.getElementById('re-shift-at').value);
    if (shiftAt > 0 && shiftAt < data.length) {
        reChartSpec.reference_lines.push({ value: shiftAt, axis: 'x', color: '#f39c12', dash: 'dotted', label: 'Shift' });
    }
    ForgeViz.render(document.getElementById('re-chart'), reChartSpec);

    // Update metrics
    const estRate = type === 'G' ? limits.pHat : limits.lambdaHat;
    document.getElementById('re-est-rate').textContent = estRate ? estRate.toFixed(4) : '—';
    document.getElementById('re-mean').textContent = limits.mean ? limits.mean.toFixed(2) : '—';
    document.getElementById('re-ooc').textContent = ooc.length;

    // Detection delay
    const shiftAtVal = parseInt(document.getElementById('re-shift-at').value);
    if (shiftAtVal > 0) {
        const firstOOCAfterShift = ooc.find(i => i >= shiftAtVal);
        if (firstOOCAfterShift !== undefined) {
            document.getElementById('re-delay').textContent = `${firstOOCAfterShift - shiftAtVal} samples`;
        } else {
            document.getElementById('re-delay').textContent = 'Not detected';
        }
    } else {
        document.getElementById('re-delay').textContent = 'N/A';
    }

    return ooc;
}

function reRenderDistribution(data, type) {
    // Histogram + fitted distribution
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const maxVal = Math.max(...data);
    const nBins = 20;

    const fitted_x = [], fitted_y = [];
    for (let i = 0; i <= 100; i++) {
        const x = maxVal * i / 100;
        fitted_x.push(x);
        if (type === 'G') {
            const p = 1 / (mean + 1);
            fitted_y.push(p * Math.pow(1 - p, x));  // Geometric PMF approx
        } else {
            const lambda = 1 / mean;
            fitted_y.push(lambda * Math.exp(-lambda * x));  // Exponential PDF
        }
    }

    // Scale fitted curve to match histogram
    const binWidth = maxVal / nBins;
    const scaleFactor = data.length * binWidth;

    // Bin the data for bar chart
    const binCounts = Array(nBins).fill(0);
    const binCenters = [];
    for (let b = 0; b < nBins; b++) {
        binCenters.push(binWidth * (b + 0.5));
    }
    data.forEach(v => {
        const b = Math.min(Math.floor(v / binWidth), nBins - 1);
        if (b >= 0) binCounts[b]++;
    });

    ForgeViz.render(document.getElementById('re-dist-chart'), {
        title: '', chart_type: 'bar',
        traces: [
            { x: binCenters, y: binCounts, name: 'Observed', trace_type: 'bar', color: 'rgba(74,159,110,0.4)' },
            { x: fitted_x, y: fitted_y.map(v => v * scaleFactor), name: type === 'G' ? 'Geometric Fit' : 'Exponential Fit', trace_type: 'line', color: '#e74c3c', width: 2 },
        ],
        x_axis: { label: type === 'G' ? 'Count Between Events' : 'Time Between Events' },
        y_axis: { label: 'Frequency' },
        reference_lines: [], markers: [], zones: [],
    });
}

function reUpdateInsights(data, ooc, type) {
    const insightSection = document.getElementById('re-insight-section');
    const insightEl = document.getElementById('re-insights');
    insightSection.style.display = 'block';

    const shiftAt = parseInt(document.getElementById('re-shift-at').value);
    const shiftMag = parseFloat(document.getElementById('re-shift-mag').value);
    const rate = parseFloat(document.getElementById('re-rate').value);
    const n = data.length;
    const mean = data.reduce((a, b) => a + b, 0) / n;

    let analysisHtml = '<div><strong style="color:var(--text-primary);">Chart Analysis</strong><br>';
    analysisHtml += `<div style="margin:4px 0;">Type: <strong>${type === 'G' ? 'G Chart (Geometric)' : 'T Chart (Exponential)'}</strong></div>`;
    analysisHtml += `<div style="margin:4px 0;">Samples: ${n}</div>`;
    analysisHtml += `<div style="margin:4px 0;">Mean between events: <strong>${mean.toFixed(2)}</strong></div>`;
    analysisHtml += `<div style="margin:4px 0;">OOC points: <strong style="color:${ooc.length > 0 ? '#e74c3c' : '#27ae60'}">${ooc.length}</strong></div>`;

    if (shiftAt > 0) {
        analysisHtml += `<br><strong>Shift Injection</strong><br>`;
        analysisHtml += `<div style="margin:4px 0;">Shift at sample ${shiftAt}: rate ${shiftMag}x (${rate} &rarr; ${(rate * shiftMag).toFixed(4)})</div>`;
        const preData = data.slice(0, shiftAt);
        const postData = data.slice(shiftAt);
        const preMean = preData.length > 0 ? preData.reduce((a, b) => a + b, 0) / preData.length : 0;
        const postMean = postData.length > 0 ? postData.reduce((a, b) => a + b, 0) / postData.length : 0;
        analysisHtml += `<div style="margin:4px 0;">Pre-shift mean: ${preMean.toFixed(2)}</div>`;
        analysisHtml += `<div style="margin:4px 0;">Post-shift mean: ${postMean.toFixed(2)} (${((postMean / preMean - 1) * 100).toFixed(0)}% change)</div>`;
        const firstOOC = ooc.find(i => i >= shiftAt);
        if (firstOOC !== undefined) {
            const delay = firstOOC - shiftAt;
            const delayColor = delay <= 3 ? '#27ae60' : delay <= 10 ? '#f39c12' : '#e74c3c';
            analysisHtml += `<div style="margin:4px 0;">Detection delay: <strong style="color:${delayColor}">${delay} samples</strong> (ARL &asymp; ${delay})</div>`;
        } else {
            analysisHtml += `<div style="margin:4px 0;color:#e74c3c;">Shift NOT detected by the chart.</div>`;
        }
    }
    analysisHtml += '</div>';

    let improvHtml = '<div><strong style="color:var(--text-primary);">Guidance</strong><br>';
    if (type === 'G') {
        improvHtml += '<div style="margin:4px 0;">The <strong>G chart</strong> monitors the number of opportunities between events. Best for rare events with constant probability (defects, adverse events, safety incidents).</div>';
        improvHtml += `<div style="margin:4px 0;">Estimated event probability: <strong>${(1/(mean+1)).toFixed(4)}</strong> (${(100/(mean+1)).toFixed(2)}%)</div>`;
        if (rate > 0.1) improvHtml += '<div style="margin:8px 0;color:#f39c12;">Event rate > 10% — consider a standard p-chart instead of G chart.</div>';
    } else {
        improvHtml += '<div style="margin:4px 0;">The <strong>T chart</strong> monitors time between events. Best for rare events measured on a time scale (equipment failures, customer complaints).</div>';
        improvHtml += `<div style="margin:4px 0;">Estimated event rate: <strong>${(1/mean).toFixed(4)}</strong> per time unit</div>`;
    }
    if (ooc.length === 0 && shiftAt > 0) {
        improvHtml += '<div style="margin:8px 0;color:#f39c12;">The shift was not detected. Try a larger shift magnitude, more samples, or supplement with CUSUM/EWMA for small shifts.</div>';
    }
    if (ooc.length > 0 && shiftAt === 0) {
        improvHtml += `<div style="margin:8px 0;color:#e74c3c;">${ooc.length} OOC point(s) detected with no injected shift — this represents false alarms at the 3-sigma level (expected ~0.27% per point).</div>`;
    }
    improvHtml += '</div>';
    insightEl.innerHTML = analysisHtml + improvHtml;
}

function reStart() {
    const mode = document.getElementById('re-mode').value;
    reState.data = reGenerateData();

    if (mode === 'instant') {
        const type = document.getElementById('re-type').value;
        const limits = reCalcLimits(reState.data, type);
        const ooc = reRenderChart(reState.data, limits, type);
        reRenderDistribution(reState.data, type);
        reUpdateInsights(reState.data, ooc, type);
    } else {
        // Simulate mode: show points one at a time
        if (reState.interval) clearInterval(reState.interval);
        reState.currentIdx = 0;
        reState.running = true;
        document.getElementById('re-speed-box').style.display = 'flex';
        document.getElementById('re-start').textContent = '⏸ Pause';
        document.getElementById('re-start').onclick = rePause;

        reState.interval = setInterval(() => {
            if (!reState.running) return;
            const speed = parseInt(document.getElementById('re-speed').value);
            for (let s = 0; s < speed; s++) {
                if (reState.currentIdx >= reState.data.length) {
                    clearInterval(reState.interval);
                    reState.running = false;
                    document.getElementById('re-start').textContent = '▶ Generate';
                    document.getElementById('re-start').onclick = reStart;
                    // Final full analysis
                    const type = document.getElementById('re-type').value;
                    const limits = reCalcLimits(reState.data, type);
                    const ooc = reRenderChart(reState.data, limits, type);
                    reRenderDistribution(reState.data, type);
                    reUpdateInsights(reState.data, ooc, type);
                    return;
                }
                reState.currentIdx++;
            }
            const shown = reState.data.slice(0, reState.currentIdx);
            const type = document.getElementById('re-type').value;
            const limits = reCalcLimits(shown, type);
            const ooc = reRenderChart(shown, limits, type);
            document.getElementById('re-progress').textContent = `Sample ${reState.currentIdx} / ${reState.data.length}`;
            if (reState.currentIdx > 10) {
                reRenderDistribution(shown, type);
                reUpdateInsights(shown, ooc, type);
            }
        }, 150);
    }
}

function rePause() {
    reState.running = !reState.running;
    document.getElementById('re-start').textContent = reState.running ? '⏸ Pause' : '▶ Resume';
}

function reReset() {
    if (reState.interval) clearInterval(reState.interval);
    reState.data = [];
    reState.running = false;
    reState.currentIdx = 0;
    document.getElementById('re-start').textContent = '▶ Generate';
    document.getElementById('re-start').onclick = reStart;
    document.getElementById('re-speed-box').style.display = 'none';
    document.getElementById('re-progress').textContent = '';
    ['re-est-rate','re-mean','re-ooc','re-delay'].forEach(id => { const el = document.getElementById(id); if (el) el.textContent = '—'; });
    const chartEl = document.getElementById('re-chart');
    if (chartEl) chartEl.innerHTML = '';
    const distEl = document.getElementById('re-dist-chart');
    if (distEl) distEl.innerHTML = '';
    document.getElementById('re-insight-section').style.display = 'none';
}

// ============================================================================
// PROBIT / DOSE-RESPONSE EXPLORER
// ============================================================================

let probitData = [];  // [{dose, n, r}]

function probitAddRow(dose='', n='', r='') {
    probitData.push({dose: +dose || 0, n: +n || 0, r: +r || 0});
    probitRenderTable();
}

function probitRemoveRow(idx) {
    probitData.splice(idx, 1);
    probitRenderTable();
}

function probitRenderTable() {
    const tbody = document.getElementById('probit-tbody');
    if (!tbody) return;
    tbody.innerHTML = probitData.map((row, i) => `
        <tr style="border-bottom:1px solid var(--border);">
            <td style="padding:6px 8px;"><input type="number" value="${row.dose}" onchange="probitData[${i}].dose=+this.value" style="width:80px;padding:4px 6px;font-size:13px;" step="any"></td>
            <td style="padding:6px 8px;"><input type="number" value="${row.n}" onchange="probitData[${i}].n=+this.value" style="width:80px;padding:4px 6px;font-size:13px;" min="1"></td>
            <td style="padding:6px 8px;"><input type="number" value="${row.r}" onchange="probitData[${i}].r=+this.value" style="width:80px;padding:4px 6px;font-size:13px;" min="0"></td>
            <td><button onclick="probitRemoveRow(${i})" style="padding:2px 8px;background:none;border:1px solid var(--border);border-radius:4px;color:var(--text-dim);cursor:pointer;">&times;</button></td>
        </tr>
    `).join('');
}

// Normal distribution helpers
function probitPhi(x) {
    return SvendMath.phi(x);
}

function probitPhiPDF(x) {
    return SvendMath.phiPDF(x);
}

function probitPhiInv(p) {
    return SvendMath.phiInv(p);
}

function probitLink(eta, model) {
    return model === 'probit' ? probitPhi(eta) : 1 / (1 + Math.exp(-eta));
}

function probitLinkDeriv(eta, model) {
    if (model === 'probit') return probitPhiPDF(eta);
    const e = Math.exp(-Math.min(eta, 500));
    return e / ((1 + e) * (1 + e));
}

function probitFit(data, model) {
    // IRLS: fit P(response) = link(a + b * log(dose))
    // Filter valid data
    const valid = data.filter(d => d.dose > 0 && d.n > 0);
    if (valid.length < 3) return null;

    const logDose = valid.map(d => Math.log(d.dose));
    const y = valid.map(d => d.r / d.n);
    const n = valid.map(d => d.n);

    // Smart initial beta via OLS on transformed proportions
    const yTrans = valid.map(d => {
        const pObs = Math.max(0.01, Math.min(0.99, d.r / d.n));
        return model === 'probit' ? probitPhiInv(pObs) : Math.log(pObs / (1 - pObs));
    });
    const xBar = logDose.reduce((a, b) => a + b, 0) / logDose.length;
    const yBar = yTrans.reduce((a, b) => a + b, 0) / yTrans.length;
    let sxy = 0, sxx = 0;
    for (let i = 0; i < logDose.length; i++) {
        sxy += (logDose[i] - xBar) * (yTrans[i] - yBar);
        sxx += (logDose[i] - xBar) * (logDose[i] - xBar);
    }
    let beta = sxx > 1e-12 ? [yBar - (sxy / sxx) * xBar, sxy / sxx] : [0, 1];
    if (beta[1] <= 0) beta = [yBar, 0.5]; // fallback: ensure positive slope
    const maxIter = 50;

    for (let iter = 0; iter < maxIter; iter++) {
        const eta = logDose.map((ld, i) => beta[0] + beta[1] * ld);
        const p = eta.map(e => probitLink(e, model));

        // Clamp p to avoid numerical issues
        const pClamped = p.map(pi => Math.max(0.001, Math.min(0.999, pi)));

        // Weights and working response for IRLS
        const deriv = eta.map(e => {
            if (model === 'probit') return probitPhiPDF(e);
            const ex = Math.exp(-Math.min(e, 500));
            return ex / ((1 + ex) * (1 + ex));
        });
        const W = pClamped.map((pi, i) => n[i] * deriv[i] * deriv[i] / (pi * (1 - pi)));
        // Clamp working response to prevent extreme values when derivative is small
        const z = eta.map((e, i) => {
            const raw = e + (y[i] - pClamped[i]) / Math.max(deriv[i], 1e-6);
            return Math.max(-20, Math.min(20, raw));
        });

        // WLS: solve (X'WX)beta = X'Wz where X = [1, logDose]
        let XtWX00 = 0, XtWX01 = 0, XtWX11 = 0, XtWz0 = 0, XtWz1 = 0;
        for (let i = 0; i < valid.length; i++) {
            const w = Math.max(W[i], 0.0001);
            XtWX00 += w;
            XtWX01 += w * logDose[i];
            XtWX11 += w * logDose[i] * logDose[i];
            XtWz0 += w * z[i];
            XtWz1 += w * logDose[i] * z[i];
        }

        const det = XtWX00 * XtWX11 - XtWX01 * XtWX01;
        if (Math.abs(det) < 1e-12) break;

        const newBeta = [
            (XtWX11 * XtWz0 - XtWX01 * XtWz1) / det,
            (-XtWX01 * XtWz0 + XtWX00 * XtWz1) / det,
        ];

        // Guard against NaN
        if (isNaN(newBeta[0]) || isNaN(newBeta[1])) break;

        const change = Math.abs(newBeta[0] - beta[0]) + Math.abs(newBeta[1] - beta[1]);
        beta = newBeta;
        if (change < 1e-8) break;
    }

    // Variance-covariance matrix (inverse of Fisher information)
    const eta = logDose.map((ld, i) => beta[0] + beta[1] * ld);
    const p = eta.map(e => probitLink(e, model));
    const pClamped = p.map(pi => Math.max(0.001, Math.min(0.999, pi)));
    const deriv = eta.map(e => {
        if (model === 'probit') return probitPhiPDF(e);
        const ex = Math.exp(-Math.min(e, 500));
        return ex / ((1 + ex) * (1 + ex));
    });
    const W = pClamped.map((pi, i) => n[i] * deriv[i] * deriv[i] / (pi * (1 - pi)));

    let I00 = 0, I01 = 0, I11 = 0;
    for (let i = 0; i < valid.length; i++) {
        const w = Math.max(W[i], 0.0001);
        I00 += w;
        I01 += w * logDose[i];
        I11 += w * logDose[i] * logDose[i];
    }
    const detI = I00 * I11 - I01 * I01;
    const vcov = detI > 1e-12 ? [
        [I11 / detI, -I01 / detI],
        [-I01 / detI, I00 / detI],
    ] : [[1,0],[0,1]];

    // Goodness of fit: Pearson chi-squared
    let chi2 = 0;
    for (let i = 0; i < valid.length; i++) {
        const expected = n[i] * pClamped[i];
        chi2 += Math.pow(valid[i].r - expected, 2) / (expected * (1 - pClamped[i]));
    }
    const df = valid.length - 2;
    // p-value from chi-squared (simple approximation)
    const chi2p = df > 0 ? 1 - probitGammaCDF(chi2 / 2, df / 2) : 1;

    return {beta, vcov, chi2, df, chi2p, valid, logDose, p: pClamped};
}

// Incomplete gamma function (for chi-squared p-value)
function probitGammaCDF(x, a) {
    if (x <= 0) return 0;
    if (a <= 0) return 1;
    // Series expansion for lower incomplete gamma
    let sum = 0, term = 1 / a;
    for (let n = 1; n < 200; n++) {
        sum += term;
        term *= x / (a + n);
        if (Math.abs(term) < 1e-12) break;
    }
    return sum * Math.exp(-x + a * Math.log(x) - lnGamma(a));
}

function lnGamma(x) {
    // Stirling's approximation
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.001208650973866179, -0.000005395239384953];
    let y = x, tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (let j = 0; j < 6; j++) ser += c[j] / ++y;
    return -tmp + Math.log(2.5066282746310005 * ser / x);
}

function probitCalcED(beta, pTarget, model) {
    // ED_p: find dose where P(response) = pTarget
    // link(a + b*log(dose)) = pTarget
    // a + b*log(dose) = link_inv(pTarget)
    if (!beta[1] || Math.abs(beta[1]) < 1e-12) return NaN;
    let etaTarget;
    if (model === 'probit') {
        etaTarget = probitPhiInv(pTarget);
    } else {
        etaTarget = Math.log(pTarget / (1 - pTarget));
    }
    const logDose = (etaTarget - beta[0]) / beta[1];
    if (logDose > 700 || logDose < -700) return NaN;
    return Math.exp(logDose);
}

function probitUpdate() {
    const valid = probitData.filter(d => d.dose > 0 && d.n > 0);
    if (valid.length < 3) return;

    const model = document.getElementById('probit-model').value;
    const confLevel = parseFloat(document.getElementById('probit-conf').value);
    const fit = probitFit(probitData, model);
    if (!fit) return;

    const {beta, vcov} = fit;

    // Calculate EDs
    const ed50 = probitCalcED(beta, 0.5, model);
    const ed10 = probitCalcED(beta, 0.1, model);
    const ed90 = probitCalcED(beta, 0.9, model);

    // ED50 CI via delta method (Fieller's theorem)
    const logED50 = (probitPhiInv(0.5) - beta[0]) / beta[1];
    // If logit, use log(0.5/0.5) = 0, so logED50 = -beta[0]/beta[1]
    const dg_da = -1 / beta[1];
    const dg_db = -(logED50) / beta[1]; // = beta[0] / beta[1]^2
    const varLogED50 = dg_da * dg_da * vcov[0][0] + 2 * dg_da * dg_db * vcov[0][1] + dg_db * dg_db * vcov[1][1];
    const zAlpha = probitPhiInv(1 - (1 - confLevel) / 2);
    const ciLow = Math.exp(logED50 - zAlpha * Math.sqrt(Math.max(0, varLogED50)));
    const ciHigh = Math.exp(logED50 + zAlpha * Math.sqrt(Math.max(0, varLogED50)));

    // Slope at ED50
    const etaAtED50 = model === 'probit' ? 0 : 0;
    const slopeAtED50 = beta[1] * (model === 'probit' ? probitPhiPDF(etaAtED50) : 0.25) / ed50;

    // Update metric cards (NaN-safe)
    const fmt = (v, d) => isNaN(v) || !isFinite(v) ? '—' : v.toFixed(d);
    document.getElementById('probit-ed50').textContent = fmt(ed50, 3);
    document.getElementById('probit-ed50-ci').textContent = `[${fmt(ciLow, 3)}, ${fmt(ciHigh, 3)}]`;
    document.getElementById('probit-ed10').textContent = fmt(ed10, 3);
    document.getElementById('probit-ed90').textContent = fmt(ed90, 3);
    document.getElementById('probit-slope').textContent = fmt(slopeAtED50, 4);
    document.getElementById('probit-gof').textContent = fmt(fit.chi2p, 4) + (fit.chi2p > 0.05 ? ' (good fit)' : ' (poor fit)');

    // Plot S-curve
    const minDose = Math.min(...valid.map(d => d.dose)) * 0.5;
    const maxDose = Math.max(...valid.map(d => d.dose)) * 1.5;
    const useLog = maxDose / minDose > 10;
    const nPts = 200;
    const curveX = [], curveY = [], curveLow = [], curveHigh = [];

    for (let i = 0; i <= nPts; i++) {
        const x = useLog
            ? Math.exp(Math.log(minDose) + (Math.log(maxDose) - Math.log(minDose)) * i / nPts)
            : minDose + (maxDose - minDose) * i / nPts;
        const eta = beta[0] + beta[1] * Math.log(x);
        const pHat = probitLink(eta, model);
        curveX.push(x);
        curveY.push(pHat);

        // Confidence band: variance of eta at this point
        const varEta = vcov[0][0] + 2 * Math.log(x) * vcov[0][1] + Math.log(x) * Math.log(x) * vcov[1][1];
        const seEta = Math.sqrt(Math.max(0, varEta));
        curveLow.push(probitLink(eta - zAlpha * seEta, model));
        curveHigh.push(probitLink(eta + zAlpha * seEta, model));
    }

    const probitSpec = {
        title: '', chart_type: 'area',
        traces: [
            // Confidence band as area
            { x: [...curveX, ...curveX.slice().reverse()], y: [...curveHigh, ...curveLow.slice().reverse()], name: `${(confLevel*100).toFixed(0)}% CI`, trace_type: 'area', color: 'rgba(74,159,110,0.15)' },
            // Fitted curve
            { x: curveX, y: curveY, name: model === 'probit' ? 'Probit Fit' : 'Logit Fit', trace_type: 'line', color: '#4a9f6e', width: 2.5 },
            // Data points
            { x: valid.map(d => d.dose), y: valid.map(d => d.r / d.n), name: 'Observed', trace_type: 'scatter', color: '#e17055', marker_size: 8 },
        ],
        reference_lines: [],
        markers: [
            { x: ed50, y: 0.5, label: `ED50 = ${ed50.toFixed(2)}`, color: '#fdcb6e' },
        ],
        zones: [],
        x_axis: { label: 'Dose' }, y_axis: { label: 'Response Probability' },
    };

    // ED reference lines
    [{ed: ed10, label: 'ED10', pct: 0.1}, {ed: ed50, label: 'ED50', pct: 0.5}, {ed: ed90, label: 'ED90', pct: 0.9}].forEach(({ed, label, pct}) => {
        if (ed > 0 && ed < maxDose * 2) {
            probitSpec.reference_lines.push({ value: ed, axis: 'x', color: '#fdcb6e', dash: 'dotted', label: label });
            probitSpec.reference_lines.push({ value: pct, axis: 'y', color: '#fdcb6e', dash: 'dotted', label: '' });
        }
    });

    ForgeViz.render(document.getElementById('probit-chart'), probitSpec);

    // Insights
    const insightSection = document.getElementById('probit-insight-section');
    const insightEl = document.getElementById('probit-insights');
    insightSection.style.display = 'block';

    let analysisHtml = '<div><strong style="color:var(--text-primary);">Model Summary</strong><br>';
    analysisHtml += `<div style="margin:4px 0;">Model: <strong>${model === 'probit' ? 'Probit' : 'Logit'}</strong></div>`;
    analysisHtml += `<div style="margin:4px 0;">Equation: P = ${model === 'probit' ? '&Phi;' : 'logistic'}(${beta[0].toFixed(3)} + ${beta[1].toFixed(3)} &times; ln(dose))</div>`;
    analysisHtml += `<div style="margin:4px 0;">Intercept (a): ${beta[0].toFixed(4)} &plusmn; ${Math.sqrt(vcov[0][0]).toFixed(4)}</div>`;
    analysisHtml += `<div style="margin:4px 0;">Slope (b): ${beta[1].toFixed(4)} &plusmn; ${Math.sqrt(vcov[1][1]).toFixed(4)}</div>`;
    analysisHtml += `<div style="margin:4px 0;">Pearson &chi;&sup2;: ${fit.chi2.toFixed(2)} (df=${fit.df}, p=${fit.chi2p.toFixed(4)})</div>`;
    const fitOk = fit.chi2p > 0.05;
    analysisHtml += `<div style="margin:4px 0;color:${fitOk ? '#27ae60' : '#e74c3c'};">${fitOk ? '&#x2705; Good fit (p > 0.05)' : '&#x274C; Poor fit (p &le; 0.05) — consider a different model or check data'}</div>`;
    analysisHtml += '</div>';

    let improvHtml = '<div><strong style="color:var(--text-primary);">Interpretation</strong><br>';
    const ratio9010 = ed90 / ed10;
    improvHtml += `<div style="margin:4px 0;">ED90/ED10 ratio: <strong>${ratio9010.toFixed(2)}</strong> — ${ratio9010 < 5 ? 'steep curve (narrow effective range)' : ratio9010 < 20 ? 'moderate slope' : 'shallow curve (wide effective range)'}</div>`;
    improvHtml += `<div style="margin:4px 0;">ED50 = ${ed50.toFixed(3)} [${ciLow.toFixed(3)}, ${ciHigh.toFixed(3)}]</div>`;
    const ciWidth = (ciHigh - ciLow) / ed50;
    if (ciWidth > 1) improvHtml += '<div style="margin:4px 0;color:#f39c12;">Wide CI on ED50 — consider adding more dose levels near the ED50 region.</div>';
    if (beta[1] < 0) improvHtml += '<div style="margin:4px 0;color:#e74c3c;">Negative slope — response DECREASES with dose. Verify data coding.</div>';
    improvHtml += '</div>';
    insightEl.innerHTML = analysisHtml + improvHtml;
}

function probitLoadExample() {
    // Classic LD50 toxicology data
    probitData = [
        {dose: 1.0, n: 50, r: 2},
        {dose: 2.0, n: 50, r: 5},
        {dose: 4.0, n: 50, r: 12},
        {dose: 8.0, n: 50, r: 25},
        {dose: 16.0, n: 50, r: 38},
        {dose: 32.0, n: 50, r: 45},
        {dose: 64.0, n: 50, r: 49},
    ];
    probitRenderTable();
    probitUpdate();
}

// ============================================================================
// MTBF / MTTR + Availability
// ============================================================================

function updateMTBFMode() {
    const mode = document.getElementById('mtbf-mode').value;
    document.getElementById('mtbf-events-section').style.display = mode === 'events' ? '' : 'none';
    document.getElementById('mtbf-direct-section').style.display = mode === 'direct' ? '' : 'none';
    if (mode === 'events') calcMTBF(); else calcMTBFDirect();
}

function calcMTBF() {
    const totalTime = parseFloat(document.getElementById('mtbf-total-time').value) || 0;
    const failures = parseFloat(document.getElementById('mtbf-failures').value) || 0;
    const downtime = parseFloat(document.getElementById('mtbf-downtime').value) || 0;

    if (failures <= 0 || totalTime <= 0) return;

    const uptime = totalTime - downtime;
    const mtbf = uptime / failures;
    const mttr = downtime / failures;
    _renderMTBF(mtbf, mttr, totalTime, failures, downtime);
}

function calcMTBFDirect() {
    const mtbf = parseFloat(document.getElementById('mtbf-direct-mtbf').value) || 0;
    const mttr = parseFloat(document.getElementById('mtbf-direct-mttr').value) || 0;
    if (mtbf <= 0) return;
    _renderMTBF(mtbf, mttr, null, null, null);
}

function _renderMTBF(mtbf, mttr, totalTime, failures, downtime) {
    const availability = mtbf / (mtbf + mttr) * 100;
    const lambda = 1 / mtbf;
    const annualDowntime = (1 - availability / 100) * 8760;

    // Nines
    let nines;
    if (availability >= 99.999) nines = 'Five nines';
    else if (availability >= 99.99) nines = 'Four nines';
    else if (availability >= 99.9) nines = 'Three nines';
    else if (availability >= 99) nines = 'Two nines';
    else if (availability >= 90) nines = 'One nine';
    else nines = 'Below one nine';

    document.getElementById('mtbf-availability').innerHTML = `${availability.toFixed(2)}<span class="calc-result-unit">%</span>`;
    document.getElementById('mtbf-result').innerHTML = `${mtbf.toFixed(1)}<span class="calc-result-unit">hrs</span>`;
    document.getElementById('mttr-result').innerHTML = `${mttr.toFixed(1)}<span class="calc-result-unit">hrs</span>`;
    document.getElementById('mtbf-lambda').innerHTML = `${lambda.toFixed(5)}<span class="calc-result-unit">/hr</span>`;
    document.getElementById('mtbf-nines').textContent = `${nines} (${availability.toFixed(3)}%)`;
    document.getElementById('mtbf-annual-downtime').textContent = `${annualDowntime.toFixed(1)} hrs/year (${(annualDowntime / 24).toFixed(1)} days)`;

    // Publish
    SvendOps.publish('mtbf', mtbf, 'hrs', 'MTBF/MTTR');
    SvendOps.publish('mttr', mttr, 'hrs', 'MTBF/MTTR');
    SvendOps.publish('availability', availability, '%', 'MTBF/MTTR');

    // Derivation
    let html = '';
    if (totalTime !== null) {
        html += `<div class="step"><div class="step-num">Step 1: MTBF</div>
            <span class="formula">MTBF = (Total Time − Downtime) / Failures</span><br>
            = (${totalTime} − ${downtime}) / ${failures} = <strong>${mtbf.toFixed(1)} hrs</strong></div>`;
        html += `<div class="step"><div class="step-num">Step 2: MTTR</div>
            <span class="formula">MTTR = Total Downtime / Failures</span><br>
            = ${downtime} / ${failures} = <strong>${mttr.toFixed(1)} hrs</strong></div>`;
    }
    html += `<div class="step"><div class="step-num">Step ${totalTime !== null ? 3 : 1}: Availability</div>
        <span class="formula">A = MTBF / (MTBF + MTTR)</span><br>
        = ${mtbf.toFixed(1)} / (${mtbf.toFixed(1)} + ${mttr.toFixed(1)}) = <strong>${availability.toFixed(3)}%</strong></div>`;
    html += `<div class="step"><div class="step-num">Step ${totalTime !== null ? 4 : 2}: Failure Rate</div>
        <span class="formula">λ = 1 / MTBF</span><br>
        = 1 / ${mtbf.toFixed(1)} = <strong>${lambda.toFixed(5)} per hour</strong></div>`;
    document.getElementById('mtbf-derivation-body').innerHTML = html;

    // Sensitivity chart: vary MTTR from 0.5x to 3x, show availability
    const mttrValues = [];
    const availValues = [];
    for (let m = Math.max(0.1, mttr * 0.2); m <= mttr * 4; m += mttr * 0.1) {
        mttrValues.push(m);
        availValues.push(mtbf / (mtbf + m) * 100);
    }

    ForgeViz.render(document.getElementById('mtbf-chart'), {
        title: '', chart_type: 'line',
        traces: [
            { x: mttrValues, y: availValues, name: 'Availability', trace_type: 'line', color: '#4a9f6e', width: 2 },
            { x: [mttr], y: [availability], name: 'Current', trace_type: 'scatter', color: '#e74c3c', marker_size: 10 },
        ],
        x_axis: { label: 'MTTR (hours)' }, y_axis: { label: 'Availability (%)' },
        reference_lines: [], markers: [], zones: [],
    });

    // Next steps
    renderNextSteps('mtbf-next-steps', [
        { title: 'OEE Analysis', desc: 'Feed availability into Overall Equipment Effectiveness', calcId: 'oee' },
        { title: 'Risk Matrix', desc: 'Assess risk of failure modes', calcId: 'riskmatrix' },
    ]);
}

// ============================================================================
// Erlang C Staffing Calculator
// ============================================================================

function _factorial(n) {
    return SvendMath.factorial(n);
}

function _erlangC(agents, traffic) {
    return SvendMath.erlangC(agents, traffic);
}

function calcErlang() {
    const arrivals = parseFloat(document.getElementById('erlang-arrivals').value) || 0;
    const serviceMin = parseFloat(document.getElementById('erlang-service').value) || 0;
    const targetTimeSec = parseFloat(document.getElementById('erlang-target-time').value) || 60;
    const targetSL = parseFloat(document.getElementById('erlang-target-sl').value) || 80;

    if (arrivals <= 0 || serviceMin <= 0) return;

    const serviceHrs = serviceMin / 60;
    const traffic = arrivals * serviceHrs; // Erlangs

    // Find minimum agents to meet SL target
    let agents = Math.ceil(traffic) + 1;
    let sl = 0, pw = 1, avgWait = Infinity;
    const targetTimeHrs = targetTimeSec / 3600;

    for (let c = Math.ceil(traffic) + 1; c <= traffic + 50; c++) {
        pw = _erlangC(c, traffic);
        sl = (1 - pw * Math.exp(-(c - traffic) * targetTimeHrs / serviceHrs)) * 100;
        avgWait = pw / (c - traffic) * serviceHrs * 3600; // seconds

        if (sl >= targetSL) {
            agents = c;
            break;
        }
        agents = c;
    }

    const occupancy = traffic / agents * 100;

    document.getElementById('erlang-agents').textContent = agents;
    document.getElementById('erlang-intensity').innerHTML = `${traffic.toFixed(2)}<span class="calc-result-unit">Erlangs</span>`;
    document.getElementById('erlang-sl').innerHTML = `${sl.toFixed(1)}<span class="calc-result-unit">%</span>`;
    document.getElementById('erlang-wait').innerHTML = `${avgWait.toFixed(0)}<span class="calc-result-unit">sec</span>`;
    document.getElementById('erlang-pw').textContent = `Probability of waiting: ${(pw * 100).toFixed(1)}%`;
    document.getElementById('erlang-occ').textContent = `Agent utilization: ${occupancy.toFixed(1)}%`;

    // Publish
    SvendOps.publish('erlang_agents', agents, 'agents', 'Erlang C');
    SvendOps.publish('erlang_sl', sl, '%', 'Erlang C');

    // Derivation
    document.getElementById('erlang-derivation-body').innerHTML = `
        <div class="step"><div class="step-num">Step 1: Traffic Intensity</div>
            <span class="formula">A = λ × S</span> (arrival rate × avg service time)<br>
            = ${arrivals} arrivals/hr × ${serviceMin} min / 60 = <strong>${traffic.toFixed(2)} Erlangs</strong></div>
        <div class="step"><div class="step-num">Step 2: Minimum Servers</div>
            Need c > A → c > ${traffic.toFixed(2)} → minimum ${Math.ceil(traffic) + 1} agents</div>
        <div class="step"><div class="step-num">Step 3: Erlang C</div>
            <span class="formula">P(wait) = [A^c/c! × c/(c−A)] / [Σ(A^k/k!) + A^c/c! × c/(c−A)]</span><br>
            With ${agents} agents: P(wait) = <strong>${(pw * 100).toFixed(1)}%</strong></div>
        <div class="step"><div class="step-num">Step 4: Service Level</div>
            <span class="formula">SL = 1 − P(wait) × e^(−(c−A) × t/S)</span><br>
            = 1 − ${(pw).toFixed(3)} × e^(−(${agents}−${traffic.toFixed(2)}) × ${targetTimeSec}s / ${(serviceMin * 60).toFixed(0)}s)<br>
            = <strong>${sl.toFixed(1)}% answered within ${targetTimeSec}s</strong></div>
        <div class="step"><div class="step-num">Step 5: Average Wait</div>
            <span class="formula">W_q = P(wait) / (c − A) × S</span><br>
            = <strong>${avgWait.toFixed(1)} seconds</strong></div>`;

    // Chart: SL vs number of agents
    const chartAgents = [], chartSL = [], chartWait = [];
    for (let c = Math.ceil(traffic) + 1; c <= agents + 6; c++) {
        const pw_ = _erlangC(c, traffic);
        const sl_ = (1 - pw_ * Math.exp(-(c - traffic) * targetTimeHrs / serviceHrs)) * 100;
        const w_ = pw_ / (c - traffic) * serviceHrs * 3600;
        chartAgents.push(c);
        chartSL.push(sl_);
        chartWait.push(w_);
    }

    ForgeViz.render(document.getElementById('erlang-chart'), {
        title: '', chart_type: 'bar',
        traces: [
            { x: chartAgents, y: chartSL, name: 'Service Level', trace_type: 'bar', color: '#4a9f6e' },
        ],
        reference_lines: [
            { value: targetSL, axis: 'y', color: '#e74c3c', dash: 'dashed', label: `Target ${targetSL}%` },
        ],
        markers: [], zones: [],
        x_axis: { label: 'Number of Agents' }, y_axis: { label: 'Service Level (%)' },
    });

    // Next steps
    renderNextSteps('erlang-next-steps', [
        { title: 'Queue Simulator', desc: 'See the queue in action with Monte Carlo variability', calcId: 'queue-sim' },
        { title: 'Cost of Quality', desc: 'Quantify the cost of wait times and abandonment', calcId: 'coq' },
    ]);
}

// ============================================================================
// Risk Matrix (5x5)
// ============================================================================

let riskRows = [
    { desc: 'Equipment failure during production', likelihood: 3, severity: 4 },
    { desc: 'Key supplier delivery delay', likelihood: 2, severity: 3 },
    { desc: 'Data breach / security incident', likelihood: 1, severity: 5 },
];

function riskAddRow() {
    riskRows.push({ desc: '', likelihood: 1, severity: 1 });
    riskRenderTable();
}

function riskDeleteRow(i) {
    riskRows.splice(i, 1);
    riskRenderTable();
    riskUpdate();
}

function riskRenderTable() {
    const tbody = document.getElementById('risk-tbody');
    tbody.innerHTML = riskRows.map((r, i) => `<tr style="border-top:1px solid var(--border);">
        <td style="padding:6px 8px;"><input type="text" value="${r.desc}" onchange="riskRows[${i}].desc=this.value"
            style="width:100%;padding:6px 8px;background:var(--bg-secondary);border:1px solid var(--border);border-radius:4px;color:var(--text-primary);font-size:13px;" placeholder="Describe the risk..."></td>
        <td style="padding:6px 8px;text-align:center;"><select onchange="riskRows[${i}].likelihood=parseInt(this.value);riskUpdate()"
            style="padding:6px;background:var(--bg-secondary);border:1px solid var(--border);border-radius:4px;color:var(--text-primary);font-size:13px;">
            ${[1,2,3,4,5].map(v => `<option value="${v}" ${r.likelihood===v?'selected':''}>${v} - ${['Rare','Unlikely','Possible','Likely','Almost Certain'][v-1]}</option>`).join('')}
            </select></td>
        <td style="padding:6px 8px;text-align:center;"><select onchange="riskRows[${i}].severity=parseInt(this.value);riskUpdate()"
            style="padding:6px;background:var(--bg-secondary);border:1px solid var(--border);border-radius:4px;color:var(--text-primary);font-size:13px;">
            ${[1,2,3,4,5].map(v => `<option value="${v}" ${r.severity===v?'selected':''}>${v} - ${['Negligible','Minor','Moderate','Major','Catastrophic'][v-1]}</option>`).join('')}
            </select></td>
        <td style="padding:6px 8px;text-align:center;font-weight:600;color:${riskColor(r.likelihood * r.severity)};">${r.likelihood * r.severity}</td>
        <td style="padding:6px 4px;text-align:center;"><button onclick="riskDeleteRow(${i})" style="background:none;border:none;color:var(--text-dim);cursor:pointer;font-size:16px;padding:2px 6px;">&times;</button></td>
    </tr>`).join('');
}

function riskColor(score) {
    if (score >= 15) return '#e74c3c';
    if (score >= 10) return '#e67e22';
    if (score >= 5) return '#f39c12';
    return '#27ae60';
}

function riskLevel(score) {
    if (score >= 15) return 'Critical';
    if (score >= 10) return 'High';
    if (score >= 5) return 'Medium';
    return 'Low';
}

function riskUpdate() {
    const total = riskRows.length;
    let high = 0, med = 0, low = 0;
    riskRows.forEach(r => {
        const s = r.likelihood * r.severity;
        if (s >= 10) high++;
        else if (s >= 5) med++;
        else low++;
    });

    document.getElementById('risk-total').textContent = total;
    document.getElementById('risk-high').textContent = high;
    document.getElementById('risk-med').textContent = med;
    document.getElementById('risk-low').textContent = low;

    // Build 5x5 heat map
    // z[severity][likelihood] = count of risks at that cell
    const z = Array.from({length: 5}, () => Array(5).fill(0));
    const annotations = [];
    const riskLabels = Array.from({length: 5}, () => Array(5).fill(''));

    riskRows.forEach(r => {
        z[r.severity - 1][r.likelihood - 1]++;
        if (riskLabels[r.severity - 1][r.likelihood - 1]) {
            riskLabels[r.severity - 1][r.likelihood - 1] += '<br>' + r.desc.substring(0, 20);
        } else {
            riskLabels[r.severity - 1][r.likelihood - 1] = r.desc.substring(0, 20);
        }
    });

    // Background risk scores (likelihood * severity)
    const bgZ = Array.from({length: 5}, (_, si) =>
        Array.from({length: 5}, (_, li) => (li + 1) * (si + 1))
    );

    const likelihoodLabels = ['1-Rare', '2-Unlikely', '3-Possible', '4-Likely', '5-Almost Certain'];
    const severityLabels = ['1-Negligible', '2-Minor', '3-Moderate', '4-Major', '5-Catastrophic'];

    // Custom colorscale: green -> yellow -> orange -> red
    const colorscale = [
        [0, '#1a3a1a'], [0.16, '#27ae60'], [0.36, '#f1c40f'],
        [0.6, '#e67e22'], [0.84, '#e74c3c'], [1, '#8b0000'],
    ];

    // Annotation text: risk score + count
    for (let si = 0; si < 5; si++) {
        for (let li = 0; li < 5; li++) {
            const score = (li + 1) * (si + 1);
            const count = z[si][li];
            let text = `<b>${score}</b>`;
            if (count > 0) text += `<br><span style="font-size:10px">${count} risk${count > 1 ? 's' : ''}</span>`;
            annotations.push({
                x: li, y: si, text: text,
                showarrow: false, font: { color: 'white', size: 12 },
            });
        }
    }

    ForgeViz.render(document.getElementById('risk-heatmap'), {
        type: 'heatmap',
        z: bgZ, x: likelihoodLabels, y: severityLabels,
        colorscale: colorscale,
        title: '', chart_type: 'scatter',
        traces: [],
        markers: annotations.map(a => ({ x: a.x, y: a.y, label: a.text, color: 'white' })),
        reference_lines: [], zones: [],
        x_axis: { label: 'Likelihood' }, y_axis: { label: 'Severity' },
    });

    // Next steps
    renderNextSteps('risk-next-steps', [
        { title: 'FMEA / RPN', desc: 'Deeper analysis with detection ratings', calcId: 'fmea' },
        { title: 'MTBF / MTTR', desc: 'Quantify failure frequency and repair time', calcId: 'mtbf' },
    ]);
}
