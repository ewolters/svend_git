// DOE Workbench — Contour, Surface, Optimizer, Desirability
// Load order: doe-state.js → doe-design.js → doe-analysis.js → doe-optimize.js → doe-power.js → doe-chat.js

// Contour model cache — enables client-side recomputation on hold slider changes
let contourModelCache = null;

function updateContourPanel() {
    if (!currentDesign || !currentAnalysis) {
        document.getElementById('no-contour-message').style.display = 'block';
        document.getElementById('contour-content').style.display = 'none';
        return;
    }

    if (currentDesign.factors.length < 2) {
        document.getElementById('no-contour-message').innerHTML =
            '<p>Contour plots require at least 2 factors.</p>';
        document.getElementById('no-contour-message').style.display = 'block';
        document.getElementById('contour-content').style.display = 'none';
        return;
    }

    document.getElementById('no-contour-message').style.display = 'none';
    document.getElementById('contour-content').style.display = 'block';

    const xSelect = document.getElementById('contour-x-factor');
    const ySelect = document.getElementById('contour-y-factor');

    const options = currentDesign.factors.map((f, i) =>
        `<option value="${f.name}" ${i === 0 ? 'selected' : ''}>${f.name}</option>`
    ).join('');

    xSelect.innerHTML = options;
    ySelect.innerHTML = currentDesign.factors.map((f, i) =>
        `<option value="${f.name}" ${i === 1 ? 'selected' : ''}>${f.name}</option>`
    ).join('');
}

async function generateContourPlot() {
    if (!currentDesign || !currentAnalysis) return;

    const xFactor = document.getElementById('contour-x-factor').value;
    const yFactor = document.getElementById('contour-y-factor').value;

    if (xFactor === yFactor) {
        alert('Please select different factors for X and Y axes.');
        return;
    }

    const results = [];
    document.querySelectorAll('.response-input').forEach(input => {
        const value = parseFloat(input.value);
        if (!isNaN(value)) {
            results.push({
                run_id: parseInt(input.dataset.runId),
                response: value,
            });
        }
    });

    const data = {
        design: currentDesign,
        results,
        x_factor: xFactor,
        y_factor: yFactor,
        hold_values: {},
        include_quadratic: document.getElementById('contour-quadratic').checked,
    };

    try {
        const response = await fetch('/api/experimenter/contour/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (response.ok) {
            renderContourFromData(result.contour, result.optimal_point);

            // Cache model for client-side recomputation
            if (result.model) {
                contourModelCache = {
                    coefficients: result.model.coefficients,
                    terms: result.model.terms,
                    factors: result.model.factors,
                    includeQuadratic: result.model.include_quadratic,
                    xFactor: xFactor,
                    yFactor: yFactor,
                };
                renderHoldSliders();
            }
        } else {
            alert(result.error || 'Failed to generate contour plot');
        }
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

// Cache last contour data for 2D/3D toggle
let lastContourData = null;
let lastContourOptimal = null;

function renderContourFromData(contour, optimalPoint) {
    lastContourData = contour;
    lastContourOptimal = optimalPoint;

    const is3D = document.getElementById('contour-3d')?.checked;
    const textColor = getComputedStyle(document.documentElement).getPropertyValue('--text-primary');

    if (is3D) {
        Plotly.react('contour-plot-container', [{
            x: contour.x,
            y: contour.y,
            z: contour.z,
            type: 'surface',
            colorscale: 'Viridis',
        }], {
            scene: {
                xaxis: { title: contour.x_label },
                yaxis: { title: contour.y_label },
                zaxis: { title: 'Response' },
                bgcolor: 'rgba(0,0,0,0)',
            },
            margin: { t: 10, r: 20, b: 40, l: 55 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: textColor },
        }, { responsive: true });
    } else {
        Plotly.react('contour-plot-container', [{
            x: contour.x,
            y: contour.y,
            z: contour.z,
            type: 'contour',
            colorscale: 'Viridis',
            contours: { coloring: 'heatmap', showlabels: true },
        }], {
            xaxis: { title: contour.x_label },
            yaxis: { title: contour.y_label },
            margin: { t: 10, r: 20, b: 40, l: 55 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: textColor },
        }, { responsive: true });
    }

    document.getElementById('contour-optimal').innerHTML = `
        <strong>Optimal Point:</strong> ${contour.x_label} = ${optimalPoint.x},
        ${contour.y_label} = ${optimalPoint.y}
        (Predicted: ${optimalPoint.z})
    `;
}

function toggle3DSurface() {
    if (lastContourData && lastContourOptimal) {
        renderContourFromData(lastContourData, lastContourOptimal);
    }
}

function renderHoldSliders() {
    const container = document.getElementById('contour-hold-sliders');
    if (!contourModelCache || !contourModelCache.factors) {
        container.style.display = 'none';
        return;
    }

    const holdFactors = contourModelCache.factors.filter(
        f => f.name !== contourModelCache.xFactor && f.name !== contourModelCache.yFactor
    );

    if (holdFactors.length === 0) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'block';
    container.innerHTML = '<label style="font-weight:600; margin-bottom:0.5rem; display:block;">Hold Values</label>' +
        holdFactors.map(f => {
            const mid = ((f.low + f.high) / 2).toFixed(2);
            return `<div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.4rem;">
                <span style="min-width:90px; font-size:0.85rem;">${f.name}</span>
                <input type="range" min="${f.low}" max="${f.high}" step="${((f.high - f.low) / 40).toFixed(4)}"
                       value="${mid}" data-factor="${f.name}" class="hold-slider"
                       oninput="onHoldSliderChange(this)" style="flex:1;">
                <span class="hold-val" data-factor="${f.name}" style="min-width:55px; text-align:right; font-size:0.85rem; font-variant-numeric:tabular-nums;">${mid}</span>
            </div>`;
        }).join('');
}

function onHoldSliderChange(slider) {
    const name = slider.dataset.factor;
    const val = parseFloat(slider.value);
    document.querySelector(`.hold-val[data-factor="${name}"]`).textContent = val.toFixed(2);
    recomputeContour();
}

function recomputeContour() {
    if (!contourModelCache) return;

    const m = contourModelCache;
    const res = 25;

    // Gather hold values (actual → coded)
    const holdCoded = {};
    document.querySelectorAll('.hold-slider').forEach(s => {
        const name = s.dataset.factor;
        const actual = parseFloat(s.value);
        const fMeta = m.factors.find(f => f.name === name);
        if (fMeta) {
            holdCoded[name] = (actual - (fMeta.low + fMeta.high) / 2) / ((fMeta.high - fMeta.low) / 2);
        }
    });

    const xMeta = m.factors.find(f => f.name === m.xFactor);
    const yMeta = m.factors.find(f => f.name === m.yFactor);
    if (!xMeta || !yMeta) return;

    // Build coded grids
    const xCoded = Array.from({ length: res }, (_, i) => -1 + (2 * i) / (res - 1));
    const yCoded = Array.from({ length: res }, (_, i) => -1 + (2 * i) / (res - 1));

    // Convert coded → actual for axis labels
    const xActual = xCoded.map(v => round4(xMeta.low + (xMeta.high - xMeta.low) * (v + 1) / 2));
    const yActual = yCoded.map(v => round4(yMeta.low + (yMeta.high - yMeta.low) * (v + 1) / 2));

    // Evaluate model over grid
    const z = [];
    let zMax = -Infinity, zMin = Infinity;
    let bestI = 0, bestJ = 0;

    for (let i = 0; i < res; i++) {
        const row = [];
        for (let j = 0; j < res; j++) {
            const factorVals = {};
            for (const f of m.factors) {
                if (f.name === m.xFactor) factorVals[f.name] = xCoded[j];
                else if (f.name === m.yFactor) factorVals[f.name] = yCoded[i];
                else factorVals[f.name] = holdCoded[f.name] || 0;
            }
            const val = evaluateModel(m.coefficients, m.terms, factorVals, m.factors);
            row.push(round4(val));
            if (val > zMax) { zMax = val; bestI = i; bestJ = j; }
            if (val < zMin) { zMin = val; }
        }
        z.push(row);
    }

    renderContourFromData(
        { x: xActual, y: yActual, z: z, x_label: m.xFactor, y_label: m.yFactor, z_min: round4(zMin), z_max: round4(zMax) },
        { x: xActual[bestJ], y: yActual[bestI], z: round4(z[bestI][bestJ]) }
    );
}

function evaluateModel(coeffs, terms, factorVals, factorMeta) {
    // Build prediction vector matching term order
    // Terms: ["Constant", "A", "B", "C", "A*B", "A*C", "B*C", "A^2", "B^2", "C^2"]
    let pred = 0;
    for (let i = 0; i < terms.length && i < coeffs.length; i++) {
        const term = terms[i];
        let val;
        if (term === 'Constant') {
            val = 1;
        } else if (term.includes('*')) {
            // Interaction: "A*B"
            const parts = term.split('*');
            val = 1;
            for (const p of parts) {
                val *= (factorVals[p.trim()] || 0);
            }
        } else if (term.includes('^')) {
            // Quadratic: "A^2"
            const base = term.split('^')[0].trim();
            const exp = parseInt(term.split('^')[1]);
            val = Math.pow(factorVals[base] || 0, exp);
        } else {
            // Main effect
            val = factorVals[term] || 0;
        }
        pred += coeffs[i] * val;
    }
    return pred;
}

function round4(v) { return Math.round(v * 10000) / 10000; }

// Optimizer — model cache for live client-side re-optimization
let optimizerModelCache = null;

let responseGoalCounter = 0;

function updateOptimizerPanel() {
    if (!currentDesign || !currentAnalysis) {
        document.getElementById('no-optimize-message').style.display = 'block';
        document.getElementById('optimize-content').style.display = 'none';
        return;
    }

    document.getElementById('no-optimize-message').style.display = 'none';
    document.getElementById('optimize-content').style.display = 'block';

    // Reset and add primary response
    responseGoalCounter = 0;
    document.getElementById('response-goals-container').innerHTML = '';
    const responseName = document.getElementById('response-name').value || 'Response';
    const overall = currentAnalysis.analysis.overall;
    _addResponseGoalCard(responseName, overall.min, overall.mean, overall.max, false);

    // Pre-cache the model from analysis coefficients for live optimization
    _buildOptimizerModel();

    const opt = currentAnalysis.optimization;
    if (opt) {
        displayOptimizationResults({
            optimization: {
                optimal_settings: opt.maximize.settings,
                predicted_responses: { [responseName]: opt.maximize.predicted },
                composite_desirability: 1.0,
            },
            interpretation: [`To maximize ${responseName}: ${Object.entries(opt.maximize.settings).map(([k,v]) => `${k}=${v}`).join(', ')}`],
        });
    }
}

function _addResponseGoalCard(name, lower, target, upper, removable) {
    const idx = responseGoalCounter++;
    const container = document.getElementById('response-goals-container');
    const card = document.createElement('div');
    card.className = 'response-goal-card';
    card.dataset.goalIdx = idx;
    card.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <h4 style="margin:0;">${removable ? `<input type="text" class="goal-name" value="${name}" placeholder="Response name" style="border:1px solid var(--border);border-radius:4px;padding:0.25rem 0.5rem;font-size:0.9rem;background:var(--bg-secondary);color:var(--text-primary);font-weight:600;width:160px;">` : name}</h4>
            ${removable ? '<button class="factor-remove" onclick="this.closest(\'.response-goal-card\').remove();">&times;</button>' : ''}
        </div>
        <div class="options-grid" style="margin-top:0.5rem;">
            <div class="form-group-inline">
                <label>Goal</label>
                <select class="goal-type">
                    <option value="maximize">Maximize</option>
                    <option value="minimize">Minimize</option>
                    <option value="target">Target</option>
                </select>
            </div>
            <div class="form-group-inline">
                <label>Lower</label>
                <input type="number" class="goal-lower" value="${lower}">
            </div>
            <div class="form-group-inline">
                <label>Target</label>
                <input type="number" class="goal-target" value="${target}">
            </div>
            <div class="form-group-inline">
                <label>Upper</label>
                <input type="number" class="goal-upper" value="${upper}">
            </div>
            <div class="form-group-inline">
                <label>Weight</label>
                <input type="range" class="goal-weight" min="0.1" max="10" step="0.1" value="1"
                       oninput="this.nextElementSibling.textContent=this.value">
                <span style="min-width:30px;text-align:right;font-size:0.85rem;">1</span>
            </div>
        </div>
    `;
    container.appendChild(card);
}

function addResponseGoal() {
    _addResponseGoalCard('Response ' + (responseGoalCounter + 1), 0, 50, 100, true);
}

function _buildOptimizerModel() {
    // Extract coefficients from analysis for client-side grid search
    if (!currentAnalysis || !currentDesign) return;

    const coeffs = currentAnalysis.analysis.coefficients;
    if (!coeffs || coeffs.length === 0) return;

    const terms = coeffs.map(c => c.term);
    const values = coeffs.map(c => c.coefficient);
    const factors = currentDesign.factors.map(f => {
        let lo = -1, hi = 1;
        try { lo = parseFloat(f.levels[0]); hi = parseFloat(f.levels[1]); } catch(e) {}
        return { name: f.name, low: lo, high: hi };
    });

    optimizerModelCache = { terms, coefficients: values, factors };
}

function _collectResponseGoals() {
    const goals = [];
    document.querySelectorAll('.response-goal-card').forEach(card => {
        const nameEl = card.querySelector('.goal-name');
        const name = nameEl ? nameEl.value.trim() : card.querySelector('h4').textContent.trim();
        goals.push({
            name: name || 'Response',
            goal: card.querySelector('.goal-type').value,
            lower: parseFloat(card.querySelector('.goal-lower').value),
            target: parseFloat(card.querySelector('.goal-target').value),
            upper: parseFloat(card.querySelector('.goal-upper').value),
            weight: parseFloat(card.querySelector('.goal-weight').value) || 1,
        });
    });
    return goals;
}

async function runOptimization() {
    if (!currentDesign || !currentAnalysis) return;

    const goals = _collectResponseGoals();
    if (goals.length === 0) return;

    // Single-response with cached model: optimize client-side (instant)
    if (goals.length === 1 && optimizerModelCache) {
        recomputeDesirability();
        return;
    }

    // Multi-response or no cache: server-side optimization
    const results = [];
    document.querySelectorAll('.response-input').forEach(input => {
        const value = parseFloat(input.value);
        if (!isNaN(value)) {
            results.push({ run_id: parseInt(input.dataset.runId), response: value });
        }
    });

    const responses = goals.map(g => ({
        name: g.name,
        results: results,
        goal: g.goal,
        lower: g.lower,
        target: g.target,
        upper: g.upper,
        weight: g.weight,
    }));

    const data = {
        design: currentDesign,
        responses: responses,
        importance: goals.map(g => g.weight),
    };

    try {
        const response = await fetch('/api/experimenter/optimize/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(data),
        });
        const result = await response.json();
        if (response.ok) {
            displayOptimizationResults(result);
        } else {
            alert(result.error || 'Optimization failed');
        }
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

function recomputeDesirability() {
    if (!optimizerModelCache) return;

    const goals = _collectResponseGoals();
    if (goals.length === 0) return;

    // Use first response for client-side single-response optimization
    const { goal, lower, target, upper, weight } = goals[0];
    const responseName = goals[0].name;

    const m = optimizerModelCache;
    const k = m.factors.length;

    // Adaptive grid resolution
    const gridPts = k <= 4 ? 11 : (k <= 5 ? 7 : 5);

    // Build grid points for each factor (-1 to 1 coded)
    const grids = m.factors.map(() =>
        Array.from({ length: gridPts }, (_, i) => -1 + (2 * i) / (gridPts - 1))
    );

    let bestD = -1, bestPred = 0, bestCombo = null;

    // Grid search — iterate all combinations
    const indices = new Array(k).fill(0);
    const total = Math.pow(gridPts, k);

    for (let iter = 0; iter < total; iter++) {
        // Build factor values from indices
        const combo = indices.map((idx, fi) => grids[fi][idx]);

        // Build factor map for evaluateModel
        const factorVals = {};
        m.factors.forEach((f, fi) => { factorVals[f.name] = combo[fi]; });

        // Predict
        const pred = evaluateOptModel(m.coefficients, m.terms, factorVals);

        // Desirability
        const d = computeDesirability(pred, goal, lower, target, upper, weight);

        if (d > bestD) {
            bestD = d;
            bestPred = pred;
            bestCombo = combo.slice();
        }

        // Increment indices (odometer)
        for (let j = k - 1; j >= 0; j--) {
            indices[j]++;
            if (indices[j] < gridPts) break;
            indices[j] = 0;
        }
    }

    // Refine around best (two-pass for large k)
    if (k > 5 && bestCombo) {
        const step = 2.0 / (gridPts - 1);
        const fineGridPts = 5;
        const fineIndices = new Array(k).fill(0);
        const fineTotal = Math.pow(fineGridPts, k);

        for (let iter = 0; iter < fineTotal; iter++) {
            const combo = fineIndices.map((idx, fi) => {
                const center = bestCombo[fi];
                return Math.max(-1, Math.min(1, center - step + (2 * step * idx) / (fineGridPts - 1)));
            });

            const factorVals = {};
            m.factors.forEach((f, fi) => { factorVals[f.name] = combo[fi]; });
            const pred = evaluateOptModel(m.coefficients, m.terms, factorVals);
            const d = computeDesirability(pred, goal, lower, target, upper, weight);

            if (d > bestD) {
                bestD = d;
                bestPred = pred;
                bestCombo = combo.slice();
            }

            for (let j = k - 1; j >= 0; j--) {
                fineIndices[j]++;
                if (fineIndices[j] < fineGridPts) break;
                fineIndices[j] = 0;
            }
        }
    }

    // Convert coded → actual
    const settings = {};
    m.factors.forEach((f, fi) => {
        if (bestCombo) {
            const actual = (f.low + f.high) / 2 + bestCombo[fi] * (f.high - f.low) / 2;
            settings[f.name] = round4(actual);
        }
    });

    displayOptimizationResults({
        optimization: {
            optimal_settings: settings,
            predicted_responses: { [responseName]: round4(bestPred) },
            composite_desirability: round4(Math.max(0, bestD)),
        },
        interpretation: [
            bestD >= 0.9 ? 'Excellent — found settings that satisfy the goal well.' :
            bestD >= 0.7 ? 'Good — settings reasonably satisfy the goal.' :
            bestD >= 0.5 ? 'Moderate — some compromise in achieving the goal.' :
            'Difficult — goal may conflict with factor constraints.',
            `Optimal: ${Object.entries(settings).map(([k,v]) => `${k}=${v}`).join(', ')}`,
        ],
    });
}

function evaluateOptModel(coeffs, terms, factorVals) {
    // Evaluate model with analysis-style term names (× for interactions, ² for quadratic)
    let pred = 0;
    for (let i = 0; i < terms.length && i < coeffs.length; i++) {
        const term = terms[i];
        let val;
        if (term === 'Constant') {
            val = 1;
        } else if (term.includes('\u00d7')) {
            // Interaction: "A×B"
            const parts = term.split('\u00d7');
            val = 1;
            for (const p of parts) val *= (factorVals[p.trim()] || 0);
        } else if (term.includes('*')) {
            // Alternative interaction syntax: "A*B"
            const parts = term.split('*');
            val = 1;
            for (const p of parts) val *= (factorVals[p.trim()] || 0);
        } else if (term.includes('\u00b2')) {
            // Quadratic: "A²"
            const base = term.replace('\u00b2', '').trim();
            val = Math.pow(factorVals[base] || 0, 2);
        } else if (term.includes('^')) {
            // Quadratic: "A^2"
            const base = term.split('^')[0].trim();
            const exp = parseInt(term.split('^')[1]);
            val = Math.pow(factorVals[base] || 0, exp);
        } else {
            val = factorVals[term] || 0;
        }
        pred += coeffs[i] * val;
    }
    return pred;
}

function computeDesirability(value, goal, lower, target, upper, weight) {
    if (goal === 'maximize') {
        if (value <= lower) return 0;
        if (value >= target) return 1;
        if (target === lower) return 1;
        return Math.pow((value - lower) / (target - lower), weight);
    } else if (goal === 'minimize') {
        if (value >= upper) return 0;
        if (value <= target) return 1;
        if (upper === target) return 1;
        return Math.pow((upper - value) / (upper - target), weight);
    } else {
        // Target
        if (value < lower || value > upper) return 0;
        if (value <= target) {
            if (target === lower) return 1;
            return Math.pow((value - lower) / (target - lower), weight);
        } else {
            if (upper === target) return 1;
            return Math.pow((upper - value) / (upper - target), weight);
        }
    }
}

function displayOptimizationResults(result) {
    document.getElementById('optimization-results').style.display = 'block';

    const opt = result.optimization;

    let settingsHtml = '<table class="data-table"><thead><tr><th>Factor</th><th>Optimal Setting</th></tr></thead><tbody>';
    for (const [factor, value] of Object.entries(opt.optimal_settings)) {
        settingsHtml += `<tr><td>${factor}</td><td class="right">${value}</td></tr>`;
    }
    settingsHtml += '</tbody></table>';

    // Predicted responses
    for (const [name, pred] of Object.entries(opt.predicted_responses)) {
        settingsHtml += `<p style="margin-top: 0.5rem;"><strong>Predicted ${name}:</strong> ${pred}</p>`;
    }

    // Individual desirability bars (when available)
    if (opt.individual_desirabilities) {
        settingsHtml += '<div style="margin-top:1rem;">';
        for (const [name, d] of Object.entries(opt.individual_desirabilities)) {
            const pct = Math.round(d * 100);
            const barColor = d >= 0.8 ? 'var(--success)' : d >= 0.5 ? 'var(--accent-primary)' : 'var(--warning)';
            settingsHtml += `
                <div style="margin-bottom:0.5rem;">
                    <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:2px;">
                        <span>${name}</span><span style="font-weight:600;">${d.toFixed(3)}</span>
                    </div>
                    <div style="height:6px;background:var(--bg-tertiary);border-radius:3px;overflow:hidden;">
                        <div style="height:100%;width:${pct}%;background:${barColor};border-radius:3px;transition:width 0.3s;"></div>
                    </div>
                </div>`;
        }
        settingsHtml += '</div>';
    }

    document.getElementById('optimal-settings-table').innerHTML = settingsHtml;
    document.getElementById('composite-desirability').textContent = opt.composite_desirability.toFixed(3);
    document.getElementById('optimization-interpretation').innerHTML =
        result.interpretation.map(i => `<p>${i}</p>`).join('');
}
