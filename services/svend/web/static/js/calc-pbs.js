/**
 * calc-pbs.js — Process Bayesian Statistics Calculators
 *
 * Load order: after calc-core.js, svend-math.js
 * Extracted from: calculators.html (inline script)
 *
 * Contains: PBS Shared Math Helpers, PBS Extended Helpers,
 *           Bayesian Cpk Calculator, Belief Chart (BOCPD),
 *           Evidence Strength (E-value), Bayesian Sigma
 */

// ============================================================================
// PBS Shared Math Helpers
// ============================================================================

function pbsParseData(text) {
    return text.split(/[\s,;\t\n]+/).map(s => s.trim()).filter(s => s !== '').map(Number).filter(v => !isNaN(v));
}

function pbsRandn() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function pbsRandChiSq(df) {
    if (df > 100) {
        const z = pbsRandn();
        const x = df * Math.pow(1 - 2/(9*df) + z * Math.sqrt(2/(9*df)), 3);
        return Math.max(x, 0.01);
    }
    let s = 0;
    for (let i = 0; i < df; i++) { const z = pbsRandn(); s += z * z; }
    return s;
}

function pbsRandInvGamma(shape, scale) {
    const x = pbsRandChiSq(2 * shape);
    return (2 * scale) / x;
}

function pbsLogsumexp(arr) {
    if (arr.length === 0) return -Infinity;
    let m = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
    if (!isFinite(m)) return m;
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += Math.exp(arr[i] - m);
    return m + Math.log(s);
}

function pbsStudentTLogPdf(x, nu, loc, scale) {
    // Log-pdf of Student-t distribution
    const z = (x - loc) / scale;
    return (lgamma((nu + 1) / 2) - lgamma(nu / 2)
            - 0.5 * Math.log(nu * Math.PI) - Math.log(scale)
            - ((nu + 1) / 2) * Math.log(1 + z * z / nu));
}

function lgamma(x) {
    // Stirling/Lanczos approximation for log(Gamma(x))
    if (x <= 0) return 0;
    // Lanczos coefficients (g=7)
    const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) {
        return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
    }
    x -= 1;
    let a = c[0];
    const t = x + 7.5;
    for (let i = 1; i < 9; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}


// ============================================================================
// PBS: Shared Extended Helpers
// ============================================================================

function pbsStudentTQuantile(p, nu) {
    // Approximate quantile of Student-t via Cornish-Fisher expansion
    // Reuses probitPhiInv for the normal quantile, then corrects for df
    const z = probitPhiInv(p);
    if (nu > 100) return z;
    const g1 = (z * z * z + z) / (4 * nu);
    const g2 = (5 * Math.pow(z, 5) + 16 * z * z * z + 3 * z) / (96 * nu * nu);
    return z + g1 + g2;
}

function pbsEstimateTrend(values, windowSize) {
    // Trailing-window OLS slope for momentum estimation
    const n = values.length;
    const w = Math.min(windowSize || 10, n);
    const slice = values.slice(n - w);
    const ybar = slice.reduce((a, b) => a + b, 0) / w;
    const xbar = (w - 1) / 2;
    let num = 0, den = 0;
    for (let i = 0; i < w; i++) {
        num += (i - xbar) * (slice[i] - ybar);
        den += (i - xbar) * (i - xbar);
    }
    return den > 0 ? num / den : 0;
}

function pbsRegimeCapability(data, alertLevels, usl, lsl, nSamples) {
    // Segment data at alarm transitions, run mini Bayesian Cpk per regime
    const regimes = [];
    let start = 0;
    for (let i = 1; i < alertLevels.length; i++) {
        if (alertLevels[i] === 'alarm' && alertLevels[i - 1] !== 'alarm') {
            regimes.push({ start, end: i - 1, data: data.slice(start, i) });
            start = i;
        }
    }
    regimes.push({ start, end: data.length - 1, data: data.slice(start) });

    const mc = nSamples || 2000;
    return regimes.filter(r => r.data.length >= 3).map((r, idx) => {
        const rd = r.data;
        const n = rd.length;
        const mean = rd.reduce((a, b) => a + b, 0) / n;
        const s2 = rd.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(n - 1, 1);
        const shape = Math.max((n - 1) / 2, 0.5);
        const scale = Math.max((n - 1) * s2 / 2, 1e-10);

        let countGt133 = 0;
        for (let i = 0; i < mc; i++) {
            const sig2 = pbsRandInvGamma(shape, scale);
            const sig = Math.sqrt(sig2);
            const mu = mean + (sig / Math.sqrt(n)) * pbsRandn();
            const cpk = Math.min((usl - mu) / (3 * sig), (mu - lsl) / (3 * sig));
            if (cpk > 1.33) countGt133++;
        }

        return {
            index: idx + 1,
            start: r.start + 1,
            end: r.end + 1,
            n,
            mean: mean.toFixed(3),
            sigma: Math.sqrt(s2).toFixed(4),
            pCpk133: (countGt133 / mc * 100).toFixed(1),
        };
    });
}


// ============================================================================
// PBS: Bayesian Cpk Calculator
// ============================================================================

function calcPbsCpk() {
    const raw = document.getElementById('pbs-cpk-data').value;
    const data = pbsParseData(raw);
    if (data.length < 5) { alert('Need at least 5 measurements.'); return; }

    const usl = parseFloat(document.getElementById('pbs-cpk-usl').value);
    const lsl = parseFloat(document.getElementById('pbs-cpk-lsl').value);
    if (isNaN(usl) || isNaN(lsl) || usl <= lsl) { alert('USL must be greater than LSL.'); return; }

    const nSamples = parseInt(document.getElementById('pbs-cpk-samples').value) || 10000;
    const targetInput = document.getElementById('pbs-cpk-target').value;
    const target = targetInput ? parseFloat(targetInput) : (usl + lsl) / 2;

    const n = data.length;
    const mean = data.reduce((a, b) => a + b, 0) / n;
    const s2 = data.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1);
    const sigma = Math.sqrt(s2);

    // Traditional Cpk
    const tradCpk = Math.min((usl - mean) / (3 * sigma), (mean - lsl) / (3 * sigma));

    // MC: sample from NIG posterior (Jeffreys prior)
    const cpkSamples = new Float64Array(nSamples);
    const defectSamples = new Float64Array(nSamples);
    const sigmaSamples = new Float64Array(nSamples);
    const ppSamples = new Float64Array(nSamples);
    const shape = (n - 1) / 2;
    const scale = (n - 1) * s2 / 2;

    for (let i = 0; i < nSamples; i++) {
        const sig2 = pbsRandInvGamma(shape, scale);
        const sig = Math.sqrt(sig2);
        const mu = mean + (sig / Math.sqrt(n)) * pbsRandn();
        cpkSamples[i] = Math.min((usl - mu) / (3 * sig), (mu - lsl) / (3 * sig));
        // Confidence Inversion: P(Y ∉ spec | θ) for each posterior draw
        defectSamples[i] = (1 - normalCDF((usl - mu) / sig)) + normalCDF((lsl - mu) / sig);
        sigmaSamples[i] = sig;
        ppSamples[i] = mu + sig * pbsRandn(); // posterior predictive Y_new
    }

    const sorted = Array.from(cpkSamples).sort((a, b) => a - b);
    const q = p => sorted[Math.min(Math.floor(p * nSamples), nSamples - 1)];
    const median = q(0.5);
    const ci025 = q(0.025);
    const ci975 = q(0.975);

    const pGt100 = sorted.filter(v => v > 1.0).length / nSamples;
    const pGt133 = sorted.filter(v => v > 1.33).length / nSamples;
    const pGt150 = sorted.filter(v => v > 1.5).length / nSamples;
    const pGt167 = sorted.filter(v => v > 1.67).length / nSamples;

    // Posterior predictive defect rate (THE confidence inversion)
    const pDefect = Array.from(defectSamples).reduce((a, b) => a + b, 0) / nSamples;
    const ppmDefect = pDefect * 1e6;

    // Posterior sigma statistics
    const sortedSigma = Array.from(sigmaSamples).sort((a, b) => a - b);
    const medianSigma = sortedSigma[Math.floor(nSamples * 0.5)];
    const sigmaCI025 = sortedSigma[Math.floor(nSamples * 0.025)];
    const sigmaCI975 = sortedSigma[Math.floor(nSamples * 0.975)];

    // Projected CI at 2x and 5x data
    const ciWidth = ci975 - ci025;
    const projCI2x = ciWidth / Math.sqrt(2);
    const projCI5x = ciWidth / Math.sqrt(5);

    // Show results
    document.getElementById('pbs-cpk-results').style.display = '';
    document.getElementById('pbs-cpk-chart-wrap').style.display = '';

    document.getElementById('pbs-cpk-trad').textContent = tradCpk.toFixed(3);
    document.getElementById('pbs-cpk-median').textContent = median.toFixed(3);
    document.getElementById('pbs-cpk-ci').textContent = `[${ci025.toFixed(3)}, ${ci975.toFixed(3)}]`;

    // Probability bars
    const probs = [
        { label: 'P(Cpk > 1.00)', value: pGt100, color: '#c0392b' },
        { label: 'P(Cpk > 1.33)', value: pGt133, color: '#e8c547' },
        { label: 'P(Cpk > 1.50)', value: pGt150, color: '#4a9f6e' },
        { label: 'P(Cpk > 1.67)', value: pGt167, color: '#3a7f8f' },
    ];
    document.getElementById('pbs-cpk-probs').innerHTML = probs.map(p => `
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
            <span style="min-width:120px; text-align:right; font-size:12px; font-family:var(--mono); color:var(--text-secondary);">${p.label}</span>
            <div style="flex:1; height:22px; background:var(--bg-secondary); border-radius:4px; overflow:hidden;">
                <div style="width:${(p.value*100).toFixed(1)}%; height:100%; background:${p.color}; border-radius:4px; display:flex; align-items:center; justify-content:flex-end; padding-right:6px; font-size:11px; font-weight:600; color:var(--bg-primary); font-family:var(--mono);">${(p.value*100).toFixed(1)}%</div>
            </div>
        </div>
    `).join('');

    // Insight
    let msg = '';
    if (pGt133 > 0.95) {
        msg = `<strong>High confidence capable.</strong> ${(pGt133*100).toFixed(0)}% probability Cpk exceeds 1.33. Credible interval: [${ci025.toFixed(3)}, ${ci975.toFixed(3)}].`;
    } else if (pGt133 > 0.5) {
        msg = `<strong>Likely capable, but uncertain.</strong> P(Cpk > 1.33) = ${(pGt133*100).toFixed(0)}%. CI: [${ci025.toFixed(3)}, ${ci975.toFixed(3)}] — more data would tighten this.`;
    } else if (pGt100 > 0.5) {
        msg = `<strong>Marginal.</strong> P(Cpk > 1.33) = ${(pGt133*100).toFixed(0)}%, P(Cpk > 1.00) = ${(pGt100*100).toFixed(0)}%. Process may be minimally capable but doesn't meet 4σ standard with confidence.`;
    } else {
        msg = `<strong>Not capable.</strong> P(Cpk > 1.00) = ${(pGt100*100).toFixed(0)}%. Posterior median Cpk = ${median.toFixed(3)}. Process needs improvement.`;
    }
    if (ciWidth > 0.5 && n < 50) {
        msg += ` <em>Wide CI (${ciWidth.toFixed(2)}) due to n=${n}. More data will sharpen the estimate.</em>`;
    }
    document.getElementById('pbs-cpk-insight').innerHTML = msg;

    // Posterior predictive display
    document.getElementById('pbs-cpk-pdefect').textContent = (pDefect * 100).toFixed(3) + '%';
    document.getElementById('pbs-cpk-ppm').textContent = ppmDefect < 1 ? '<1' : Math.round(ppmDefect).toLocaleString();
    document.getElementById('pbs-cpk-sigma').textContent =
        `${medianSigma.toFixed(4)} [${sigmaCI025.toFixed(4)}, ${sigmaCI975.toFixed(4)}]`;

    // Data projection
    const projEl = document.getElementById('pbs-cpk-projection');
    projEl.style.display = '';
    projEl.innerHTML = `<strong>Data projection:</strong> With 2&times; data (n=${2*n}), Cpk CI narrows to &plusmn;${(projCI2x/2).toFixed(3)}. ` +
        `With 5&times; data (n=${5*n}), CI narrows to &plusmn;${(projCI5x/2).toFixed(3)}. ` +
        `<span style="color:var(--text-dim);">Current CI width: ${ciWidth.toFixed(3)}, n=${n}.</span>`;

    // Plotly histogram
    const nBins = 60;
    const lo = sorted[Math.floor(sorted.length * 0.005)];
    const hi = sorted[Math.floor(sorted.length * 0.995)];
    const filtered = sorted.filter(v => v >= lo && v <= hi);

    Plotly.newPlot('pbs-cpk-chart', [{
        x: filtered, type: 'histogram', nbinsx: nBins,
        marker: { color: 'rgba(58,127,143,0.5)', line: { color: 'rgba(58,127,143,0.8)', width: 1 } },
        hovertemplate: 'Cpk: %{x:.3f}<br>Count: %{y}<extra></extra>',
    }], {
        shapes: [
            { type: 'line', x0: 1.0, x1: 1.0, y0: 0, y1: 1, yref: 'paper', line: { color: '#c0392b', width: 2, dash: 'dash' } },
            { type: 'line', x0: 1.33, x1: 1.33, y0: 0, y1: 1, yref: 'paper', line: { color: '#e8c547', width: 2, dash: 'dash' } },
            { type: 'line', x0: 1.67, x1: 1.67, y0: 0, y1: 1, yref: 'paper', line: { color: '#4a9f6e', width: 2, dash: 'dash' } },
            { type: 'line', x0: median, x1: median, y0: 0, y1: 1, yref: 'paper', line: { color: '#3a7f8f', width: 2.5 } },
        ],
        annotations: [
            { x: 1.0, y: 1, yref: 'paper', text: '1.00', showarrow: false, yanchor: 'bottom', font: { size: 10, color: '#c0392b' } },
            { x: 1.33, y: 1, yref: 'paper', text: '1.33', showarrow: false, yanchor: 'bottom', font: { size: 10, color: '#e8c547' } },
            { x: 1.67, y: 1, yref: 'paper', text: '1.67', showarrow: false, yanchor: 'bottom', font: { size: 10, color: '#4a9f6e' } },
            { x: median, y: 1, yref: 'paper', text: 'Median', showarrow: false, yanchor: 'bottom', font: { size: 10, color: '#3a7f8f' } },
        ],
        xaxis: { title: 'Cpk', color: '#9aaa9a' },
        yaxis: { title: 'Count', color: '#9aaa9a' },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a' }, margin: { t: 30, r: 20, b: 50, l: 50 },
    }, { responsive: true, displayModeBar: false });

    // Posterior predictive histogram — "Where the next part falls"
    document.getElementById('pbs-cpk-pp-chart-wrap').style.display = '';
    const ppSorted = Array.from(ppSamples).sort((a, b) => a - b);
    const ppLo = ppSorted[Math.floor(ppSorted.length * 0.005)];
    const ppHi = ppSorted[Math.floor(ppSorted.length * 0.995)];
    const ppFiltered = ppSorted.filter(v => v >= ppLo && v <= ppHi);

    Plotly.newPlot('pbs-cpk-pp-chart', [{
        x: ppFiltered, type: 'histogram', nbinsx: 80,
        marker: { color: 'rgba(74,159,110,0.4)', line: { color: 'rgba(74,159,110,0.7)', width: 1 } },
        hovertemplate: 'Value: %{x:.3f}<br>Count: %{y}<extra></extra>',
    }], {
        shapes: [
            { type: 'line', x0: usl, x1: usl, y0: 0, y1: 1, yref: 'paper',
              line: { color: '#c0392b', width: 2.5 } },
            { type: 'line', x0: lsl, x1: lsl, y0: 0, y1: 1, yref: 'paper',
              line: { color: '#c0392b', width: 2.5 } },
            { type: 'line', x0: target, x1: target, y0: 0, y1: 1, yref: 'paper',
              line: { color: '#3a7f8f', width: 1.5, dash: 'dash' } },
        ],
        annotations: [
            { x: usl, y: 1, yref: 'paper', text: 'USL', showarrow: false,
              yanchor: 'bottom', font: { size: 11, color: '#c0392b' } },
            { x: lsl, y: 1, yref: 'paper', text: 'LSL', showarrow: false,
              yanchor: 'bottom', font: { size: 11, color: '#c0392b' } },
        ],
        xaxis: { title: 'Predicted Value (Y_new)', color: '#9aaa9a' },
        yaxis: { title: 'Count', color: '#9aaa9a' },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a' }, margin: { t: 30, r: 20, b: 50, l: 50 },
    }, { responsive: true, displayModeBar: false });

    // Publish to SvendOps data bus
    SvendOps.publish('pbsUSL', usl, '', 'Bayesian Cpk');
    SvendOps.publish('pbsLSL', lsl, '', 'Bayesian Cpk');
    SvendOps.publish('pbsCpkMedian', parseFloat(median.toFixed(3)), '', 'Bayesian Cpk');
    SvendOps.publish('pbsPDefect', parseFloat((pDefect * 100).toFixed(3)), '%', 'Bayesian Cpk');
    SvendOps.publish('pbsSigma', parseFloat(medianSigma.toFixed(4)), '', 'Bayesian Cpk');

    // Next steps
    document.getElementById('pbs-cpk-next-steps').style.display = '';
    renderNextSteps('pbs-cpk-next-steps', [
        { title: 'Belief Chart', desc: 'Detect shifts — pulls your spec limits', calcId: 'pbs-belief' },
        { title: 'Evidence Strength', desc: 'Sequential evidence against a reference', calcId: 'pbs-evidence' },
        { title: 'Bayesian Σ', desc: 'Escape probability analysis', calcId: 'pbs-sigma' },
    ]);
}


// ============================================================================
// PBS: Belief Chart (BOCPD) Calculator
// ============================================================================

function calcPbsBelief() {
    const raw = document.getElementById('pbs-belief-data').value;
    const data = pbsParseData(raw);
    if (data.length < 5) { alert('Need at least 5 observations.'); return; }

    const hazardLambda = parseFloat(document.getElementById('pbs-belief-hazard').value) || 200;
    const refInput = document.getElementById('pbs-belief-ref').value;

    const n = data.length;
    const H = 1.0 / hazardLambda;
    const logH = Math.log(H);
    const log1mH = Math.log(1.0 - H);
    const K = 200; // max run lengths

    // Prior: vague Normal-Gamma
    const prior = { mu: refInput ? parseFloat(refInput) : data[0], kappa: 0.01, alpha: 0.01, beta: 0.01 };

    // State
    let logR = [0.0]; // log run length distribution
    let suff = [{ mu: prior.mu, kappa: prior.kappa, alpha: prior.alpha, beta: prior.beta }];

    const shiftProbs = [];
    const regimeMeans = [];
    const regimeStds = [];
    const regimeMeanUpper = [];
    const regimeMeanLower = [];
    const regimeKappas = [];
    const alertLevels = [];

    for (let t = 0; t < n; t++) {
        const x = data[t];
        const nR = logR.length;

        // 1. Predictive log-pdf under each run length
        const logPred = new Float64Array(nR);
        for (let i = 0; i < nR; i++) {
            const s = suff[i];
            const nu = 2 * s.alpha;
            const loc = s.mu;
            const scale = Math.sqrt(s.beta * (s.kappa + 1) / (s.alpha * s.kappa));
            logPred[i] = pbsStudentTLogPdf(x, nu, loc, scale);
        }

        // 2. Growth probabilities
        const logGrowth = new Float64Array(nR);
        for (let i = 0; i < nR; i++) {
            logGrowth[i] = logR[i] + logPred[i] + log1mH;
        }

        // 3. Changepoint probability
        const cpTerms = new Float64Array(nR);
        for (let i = 0; i < nR; i++) {
            cpTerms[i] = logR[i] + logPred[i] + logH;
        }
        const logCp = pbsLogsumexp(Array.from(cpTerms));

        // 4. Combine
        const newLogR = new Float64Array(nR + 1);
        newLogR[0] = logCp;
        for (let i = 0; i < nR; i++) newLogR[i + 1] = logGrowth[i];

        // 5. Normalize
        const logEvidence = pbsLogsumexp(Array.from(newLogR));
        for (let i = 0; i < newLogR.length; i++) newLogR[i] -= logEvidence;

        // 6. Update sufficient statistics
        const newSuff = [{ mu: prior.mu, kappa: prior.kappa, alpha: prior.alpha, beta: prior.beta }];
        for (let i = 0; i < nR; i++) {
            const s = suff[i];
            const kNew = s.kappa + 1;
            const muNew = (s.kappa * s.mu + x) / kNew;
            const aNew = s.alpha + 0.5;
            const bNew = s.beta + s.kappa * (x - s.mu) ** 2 / (2 * kNew);
            newSuff.push({ mu: muNew, kappa: Math.min(kNew, 1e8), alpha: aNew, beta: Math.max(bNew, 1e-15) });
        }

        // 7. Truncate to top K
        let finalLogR = Array.from(newLogR);
        let finalSuff = newSuff;
        if (finalLogR.length > K) {
            const indices = Array.from({ length: finalLogR.length }, (_, i) => i);
            indices.sort((a, b) => finalLogR[b] - finalLogR[a]);
            const topK = indices.slice(0, K).sort((a, b) => a - b);
            finalLogR = topK.map(i => finalLogR[i]);
            finalSuff = topK.map(i => finalSuff[i]);
            const norm = pbsLogsumexp(finalLogR);
            for (let i = 0; i < finalLogR.length; i++) finalLogR[i] -= norm;
        }

        logR = finalLogR;
        suff = finalSuff;

        // 8. Shift probability = 1 - P(longest run)
        const sp = logR.length > 1 ? Math.max(0, Math.min(1, 1 - Math.exp(logR[logR.length - 1]))) : 0;
        shiftProbs.push(sp);

        // 9. MAP run length regime parameters
        let mlIdx = 0;
        for (let i = 1; i < logR.length; i++) if (logR[i] > logR[mlIdx]) mlIdx = i;
        const ms = suff[mlIdx];
        regimeMeans.push(ms.mu);
        regimeStds.push(Math.sqrt(ms.beta / Math.max(ms.alpha - 1, 0.5)));

        // 10. Credible interval on regime mean (Student-t from Normal-Gamma posterior)
        const nu_t = 2 * ms.alpha;
        const scale_t = Math.sqrt(ms.beta / (ms.alpha * Math.max(ms.kappa, 0.01)));
        const tq975 = pbsStudentTQuantile(0.975, Math.max(nu_t, 1));
        regimeMeanUpper.push(ms.mu + tq975 * scale_t);
        regimeMeanLower.push(ms.mu - tq975 * scale_t);
        regimeKappas.push(ms.kappa);

        // Alert level
        if (sp >= 0.95) alertLevels.push('alarm');
        else if (sp >= 0.80) alertLevels.push('alert');
        else if (sp >= 0.50) alertLevels.push('watch');
        else alertLevels.push('nominal');
    }

    // Count shifts (transitions from nominal to alarm)
    let shiftCount = 0;
    for (let i = 1; i < alertLevels.length; i++) {
        if (alertLevels[i] === 'alarm' && alertLevels[i - 1] !== 'alarm') shiftCount++;
    }
    const maxProb = Math.max(...shiftProbs);

    // Show results
    document.getElementById('pbs-belief-results').style.display = '';
    document.getElementById('pbs-belief-chart-wrap').style.display = '';

    document.getElementById('pbs-belief-shifts').textContent = shiftCount;
    document.getElementById('pbs-belief-maxprob').textContent = (maxProb * 100).toFixed(1) + '%';
    document.getElementById('pbs-belief-regime-mean').textContent = regimeMeans[n - 1].toFixed(3);
    document.getElementById('pbs-belief-regime-std').textContent = regimeStds[n - 1].toFixed(4);
    document.getElementById('pbs-belief-neff').textContent = regimeKappas[n - 1].toFixed(1);
    document.getElementById('pbs-belief-mean-ci').textContent =
        `[${regimeMeanLower[n - 1].toFixed(3)}, ${regimeMeanUpper[n - 1].toFixed(3)}]`;

    // Two-panel Plotly chart
    const tArr = data.map((_, i) => i + 1);

    // Color shift probability bars by alert level
    const spColors = alertLevels.map(a =>
        a === 'alarm' ? '#c0392b' : a === 'alert' ? '#e8c547' : a === 'watch' ? '#e8a547' : 'rgba(58,127,143,0.5)'
    );

    Plotly.newPlot('pbs-belief-chart', [
        // Top: credible band on regime mean
        { x: tArr, y: regimeMeanUpper, mode: 'lines', line: { width: 0 },
          showlegend: false, yaxis: 'y1', hoverinfo: 'skip' },
        { x: tArr, y: regimeMeanLower, mode: 'lines', line: { width: 0 },
          fill: 'tonexty', fillcolor: 'rgba(232,197,71,0.15)',
          showlegend: false, yaxis: 'y1', name: '95% CI',
          hovertemplate: 't=%{x}<br>95% CI lower: %{y:.3f}<extra></extra>' },
        // Top: observations + regime mean
        { x: tArr, y: data, mode: 'markers', marker: { size: 5, color: 'rgba(58,127,143,0.7)' },
          name: 'Observations', yaxis: 'y1', hovertemplate: 't=%{x}<br>Value: %{y:.3f}<extra></extra>' },
        { x: tArr, y: regimeMeans, mode: 'lines', line: { color: '#e8c547', width: 2 },
          name: 'Regime Mean', yaxis: 'y1', hovertemplate: 't=%{x}<br>Regime μ: %{y:.3f}<extra></extra>' },
        // Bottom: shift probability
        { x: tArr, y: shiftProbs, type: 'bar', marker: { color: spColors },
          name: 'Shift Prob', yaxis: 'y2', hovertemplate: 't=%{x}<br>P(shift): %{y:.3f}<extra></extra>' },
    ], {
        grid: { rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
        yaxis: { title: 'Value', domain: [0.45, 1.0], color: '#9aaa9a' },
        yaxis2: { title: 'P(shift)', domain: [0.0, 0.38], range: [0, 1.05], color: '#9aaa9a' },
        xaxis: { visible: false },
        xaxis2: { title: 'Observation #', color: '#9aaa9a' },
        shapes: [
            { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 0.50, y1: 0.50, yref: 'y2', line: { color: '#e8a547', width: 1, dash: 'dot' } },
            { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 0.80, y1: 0.80, yref: 'y2', line: { color: '#e8c547', width: 1, dash: 'dot' } },
            { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 0.95, y1: 0.95, yref: 'y2', line: { color: '#c0392b', width: 1, dash: 'dot' } },
        ],
        showlegend: false,
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a' }, margin: { t: 20, r: 20, b: 50, l: 60 },
    }, { responsive: true, displayModeBar: false });

    // Publish to SvendOps data bus
    SvendOps.publish('pbsCurrentRegimeMean', parseFloat(regimeMeans[n-1].toFixed(3)), '', 'Belief Chart');
    SvendOps.publish('pbsCurrentRegimeSigma', parseFloat(regimeStds[n-1].toFixed(4)), '', 'Belief Chart');
    SvendOps.publish('pbsRegimeNeff', parseFloat(regimeKappas[n-1].toFixed(1)), '', 'Belief Chart');
    SvendOps.publish('pbsShiftCount', shiftCount, 'shifts', 'Belief Chart');

    // Per-regime capability (if spec limits available)
    const beliefUSL = parseFloat(document.getElementById('pbs-belief-usl').value);
    const beliefLSL = parseFloat(document.getElementById('pbs-belief-lsl').value);
    const capSection = document.getElementById('pbs-belief-capability-section');
    if (!isNaN(beliefUSL) && !isNaN(beliefLSL) && beliefUSL > beliefLSL) {
        const regimes = pbsRegimeCapability(data, alertLevels, beliefUSL, beliefLSL);
        capSection.style.display = '';
        document.getElementById('pbs-belief-regime-table').innerHTML = `
            <table style="width:100%; border-collapse:collapse; font-size:12px; font-family:var(--mono);">
                <thead><tr style="border-bottom:1px solid var(--border); color:var(--text-secondary);">
                    <th style="padding:6px; text-align:left;">Regime</th>
                    <th style="padding:6px;">Range</th>
                    <th style="padding:6px;">n</th>
                    <th style="padding:6px;">Mean</th>
                    <th style="padding:6px;">&sigma;</th>
                    <th style="padding:6px;">P(Cpk&gt;1.33)</th>
                </tr></thead>
                <tbody>${regimes.map(r => `
                    <tr style="border-bottom:1px solid rgba(150,150,150,0.1);">
                        <td style="padding:6px;">${r.index}</td>
                        <td style="padding:6px; text-align:center;">t=${r.start}\u2013${r.end}</td>
                        <td style="padding:6px; text-align:center;">${r.n}</td>
                        <td style="padding:6px; text-align:center;">${r.mean}</td>
                        <td style="padding:6px; text-align:center;">${r.sigma}</td>
                        <td style="padding:6px; text-align:center; font-weight:600;
                            color:${parseFloat(r.pCpk133) > 90 ? '#4a9f6e' : parseFloat(r.pCpk133) > 50 ? '#e8c547' : '#c0392b'};">
                            ${r.pCpk133}%</td>
                    </tr>`).join('')}
                </tbody>
            </table>`;
    } else {
        capSection.style.display = 'none';
    }

    // Next steps
    document.getElementById('pbs-belief-next-steps').style.display = '';
    renderNextSteps('pbs-belief-next-steps', [
        { title: 'Bayesian Cpk', desc: 'Capability on current regime', calcId: 'pbs-cpk' },
        { title: 'Evidence Strength', desc: 'Test against pre-shift mean', calcId: 'pbs-evidence' },
        { title: 'Bayesian Σ', desc: 'Escape probability analysis', calcId: 'pbs-sigma' },
    ]);
}


// ============================================================================
// PBS: Evidence Strength (E-value) Calculator
// ============================================================================

function calcPbsEvidence() {
    const raw = document.getElementById('pbs-evidence-data').value;
    const data = pbsParseData(raw);
    if (data.length < 5) { alert('Need at least 5 observations.'); return; }

    const n = data.length;

    // Reference parameters — use first 20% as calibration if not specified
    const calN = Math.max(3, Math.floor(n * 0.2));
    const muInput = document.getElementById('pbs-evidence-mu').value;
    const sigInput = document.getElementById('pbs-evidence-sigma').value;

    let mu0, sigma;
    if (muInput) {
        mu0 = parseFloat(muInput);
    } else {
        mu0 = data.slice(0, calN).reduce((a, b) => a + b, 0) / calN;
    }
    if (sigInput) {
        sigma = parseFloat(sigInput);
    } else {
        const calData = data.slice(0, calN);
        const calMean = calData.reduce((a, b) => a + b, 0) / calN;
        sigma = Math.sqrt(calData.reduce((s, v) => s + (v - calMean) ** 2, 0) / Math.max(calN - 1, 1));
        if (sigma < 1e-10) sigma = 0.01; // fallback for constant data
    }

    const sigma2 = sigma * sigma;
    const sigma2Mix = sigma2 * 1.0; // sigma_mix_ratio = 1.0

    // Accumulate e-values (Grünwald 2024)
    let logE = 0;
    const eValues = [];
    const logEs = [];
    const levels = [];

    for (let t = 0; t < n; t++) {
        const x = data[t];
        const ratio = sigma2 / (sigma2 + sigma2Mix);
        const exponent = sigma2Mix * (x - mu0) ** 2 / (2 * sigma2 * (sigma2 + sigma2Mix));
        const logEi = 0.5 * Math.log(Math.max(ratio, 1e-300)) + exponent;

        logE += logEi;

        // Cap display at 10000
        const logEDisplay = Math.min(logE, Math.log(10000));
        const eDisplay = Math.exp(logEDisplay);

        eValues.push(eDisplay);
        logEs.push(logE);

        if (eDisplay >= 100) levels.push('decisive');
        else if (eDisplay >= 20) levels.push('strong');
        else if (eDisplay >= 5) levels.push('notable');
        else levels.push('none');
    }

    const finalE = eValues[n - 1];
    const peakE = Math.max(...eValues);
    const finalLevel = levels[n - 1];

    // Posterior odds (E-value IS the likelihood ratio)
    const pShifted = finalE / (finalE + 1); // P(shifted | data) with 50/50 prior

    // Show results
    document.getElementById('pbs-evidence-results').style.display = '';
    document.getElementById('pbs-evidence-chart-wrap').style.display = '';

    document.getElementById('pbs-evidence-final').textContent = finalE >= 100 ? finalE.toFixed(0) : finalE.toFixed(2);
    document.getElementById('pbs-evidence-peak').textContent = peakE >= 100 ? peakE.toFixed(0) : peakE.toFixed(2);

    const levelColors = { decisive: '#c0392b', strong: '#e8c547', notable: '#4a9f6e', none: '#9aaa9a' };
    const levelEl = document.getElementById('pbs-evidence-level');
    levelEl.textContent = finalLevel.charAt(0).toUpperCase() + finalLevel.slice(1);
    levelEl.style.color = levelColors[finalLevel];

    document.getElementById('pbs-evidence-pshifted').textContent = (pShifted * 100).toFixed(1) + '%';

    // Insight
    let msg = '';
    const oddsStr = finalE >= 100 ? finalE.toFixed(0) : finalE.toFixed(1);
    if (finalLevel === 'decisive') {
        msg = `<strong>Decisive evidence</strong> against H₀: μ = ${mu0.toFixed(2)}. The e-value of ${finalE.toFixed(1)} means data are ${finalE.toFixed(0)}× more likely under a shifted process. Posterior P(shifted) = ${(pShifted*100).toFixed(1)}% (odds ${oddsStr}:1). This is anytime-valid — no correction for multiple looks.`;
    } else if (finalLevel === 'strong') {
        msg = `<strong>Strong evidence</strong> of a shift from μ₀ = ${mu0.toFixed(2)}. E-value = ${finalE.toFixed(1)} exceeds the strong threshold (20). Posterior P(shifted) = ${(pShifted*100).toFixed(1)}% (odds ${oddsStr}:1). Continue monitoring or investigate.`;
    } else if (finalLevel === 'notable') {
        msg = `<strong>Notable evidence</strong> accumulating against μ₀ = ${mu0.toFixed(2)}. E-value = ${finalE.toFixed(2)} suggests something may be changing. Posterior P(shifted) = ${(pShifted*100).toFixed(1)}% (odds ${oddsStr}:1). More data needed for strong conclusion.`;
    } else {
        msg = `<strong>No significant evidence</strong> against H₀: μ = ${mu0.toFixed(2)}. E-value = ${finalE.toFixed(2)} is below the notable threshold (5). Posterior P(shifted) = ${(pShifted*100).toFixed(1)}%. The process appears stable relative to reference.`;
    }
    msg += ` <br><span style="font-size:11px; color:var(--text-dim);">Reference: μ₀=${mu0.toFixed(3)}, σ=${sigma.toFixed(4)}. Calibration: first ${calN} observations${muInput ? ' (μ₀ user-specified)' : ''}${sigInput ? ' (σ user-specified)' : ''}.</span>`;
    document.getElementById('pbs-evidence-insight').innerHTML = msg;

    // Plotly chart
    const tArr = data.map((_, i) => i + 1);

    Plotly.newPlot('pbs-evidence-chart', [
        { x: tArr, y: eValues, mode: 'lines+markers',
          line: { color: '#3a7f8f', width: 2 },
          marker: { size: 4, color: levels.map(l => levelColors[l]) },
          name: 'E-value',
          hovertemplate: 't=%{x}<br>E-value: %{y:.2f}<extra></extra>' },
    ], {
        shapes: [
            { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 5, y1: 5, line: { color: '#4a9f6e', width: 1, dash: 'dot' } },
            { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 20, y1: 20, line: { color: '#e8c547', width: 1, dash: 'dot' } },
            { type: 'line', x0: 0, x1: 1, xref: 'paper', y0: 100, y1: 100, line: { color: '#c0392b', width: 1, dash: 'dot' } },
        ],
        annotations: [
            { x: 1, xref: 'paper', y: 5, text: 'Notable (5)', showarrow: false, xanchor: 'left', font: { size: 10, color: '#4a9f6e' } },
            { x: 1, xref: 'paper', y: 20, text: 'Strong (20)', showarrow: false, xanchor: 'left', font: { size: 10, color: '#e8c547' } },
            { x: 1, xref: 'paper', y: 100, text: 'Decisive (100)', showarrow: false, xanchor: 'left', font: { size: 10, color: '#c0392b' } },
        ],
        xaxis: { title: 'Observation #', color: '#9aaa9a' },
        yaxis: { title: 'E-value', color: '#9aaa9a', type: peakE > 50 ? 'log' : 'linear' },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a' }, margin: { t: 20, r: 80, b: 50, l: 60 },
    }, { responsive: true, displayModeBar: false });

    // Publish to SvendOps data bus
    SvendOps.publish('pbsEvalue', parseFloat(finalE >= 100 ? finalE.toFixed(0) : finalE.toFixed(2)), '', 'Evidence Strength');
    SvendOps.publish('pbsReferenceMu', parseFloat(mu0.toFixed(3)), '', 'Evidence Strength');
    SvendOps.publish('pbsEvidenceLevel', finalLevel, '', 'Evidence Strength');

    // Next steps
    document.getElementById('pbs-evidence-next-steps').style.display = '';
    renderNextSteps('pbs-evidence-next-steps', [
        { title: 'Belief Chart', desc: 'Pinpoint where shifts occurred', calcId: 'pbs-belief' },
        { title: 'Bayesian Cpk', desc: 'Quantify capability with uncertainty', calcId: 'pbs-cpk' },
        { title: 'Bayesian Σ', desc: 'Escape probability analysis', calcId: 'pbs-sigma' },
    ]);
}


// ============================================================================
// PBS: Bayesian Sigma Calculator
// ============================================================================

function calcPbsSigma() {
    const raw = document.getElementById('pbs-sigma-data').value;
    const data = pbsParseData(raw);
    if (data.length < 5) { alert('Need at least 5 observations.'); return; }

    const usl = parseFloat(document.getElementById('pbs-sigma-usl').value);
    const lsl = parseFloat(document.getElementById('pbs-sigma-lsl').value);
    if (isNaN(usl) || isNaN(lsl) || usl <= lsl) { alert('USL must be greater than LSL.'); return; }

    const nSamples = parseInt(document.getElementById('pbs-sigma-samples').value) || 10000;
    const trendWindow = parseInt(document.getElementById('pbs-sigma-window').value) || 10;

    const n = data.length;
    const mean = data.reduce((a, b) => a + b, 0) / n;
    const s2 = data.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1);
    const sigma = Math.sqrt(s2);

    // Trend (momentum) estimation via trailing-window OLS
    const trend = pbsEstimateTrend(data, trendWindow);

    // Nearest spec limit direction
    const distToUSL = usl - mean;
    const distToLSL = mean - lsl;
    const nearestIsUSL = distToUSL < distToLSL;

    // Momentum alignment: positive = trending toward nearest spec
    const alignmentRaw = nearestIsUSL ? trend : -trend;
    // Normalize to [-1, 1] via tanh
    const alignmentNorm = Math.tanh(alignmentRaw / (sigma / Math.sqrt(trendWindow) + 1e-10));
    const alignmentPlus = Math.max(0, alignmentNorm); // A+ = max(0, A)

    // NIG posterior (Jeffreys prior)
    const shape = (n - 1) / 2;
    const scale = (n - 1) * s2 / 2;

    // MC sampling
    const sigmaBSamples = new Float64Array(nSamples);
    const escapeSamples = new Float64Array(nSamples);
    const T_escape = 100; // time horizon for escape probability

    for (let i = 0; i < nSamples; i++) {
        const sig2 = pbsRandInvGamma(shape, scale);
        const sig = Math.sqrt(sig2);
        const mu = mean + (sig / Math.sqrt(n)) * pbsRandn();

        // Quasipotential: V = d_nearest² / 2
        const dUSL = usl - mu;
        const dLSL = mu - lsl;
        const dNearest = Math.min(Math.abs(dUSL), Math.abs(dLSL));
        const V = dNearest * dNearest / 2;
        const epsilon = sig2; // noise intensity

        // Bayesian Sigma: Σ_B = sqrt(2V/ε) · sqrt(1 - A+)
        // When A+=0: sqrt(2V/ε) = sqrt(d²/σ²) = d/σ — classical sigma metric
        const baseSigma = Math.sqrt(2 * V / Math.max(epsilon, 1e-15));
        const momentumFactor = Math.sqrt(Math.max(1 - alignmentPlus, 0.01));
        sigmaBSamples[i] = baseSigma * momentumFactor;

        // Kramers escape: P(escape in T) ≈ 1 - exp(-T · exp(-2V/ε))
        const kramersRate = Math.exp(-2 * V / Math.max(epsilon, 1e-15));
        escapeSamples[i] = 1 - Math.exp(-T_escape * kramersRate);
    }

    const sorted = Array.from(sigmaBSamples).sort((a, b) => a - b);
    const q = p => sorted[Math.min(Math.floor(p * nSamples), nSamples - 1)];
    const median = q(0.5);
    const ci025 = q(0.025);
    const ci975 = q(0.975);

    const pGt2 = sorted.filter(v => v > 2).length / nSamples;
    const pGt3 = sorted.filter(v => v > 3).length / nSamples;
    const pGt4 = sorted.filter(v => v > 4).length / nSamples;
    const pGt6 = sorted.filter(v => v > 6).length / nSamples;
    const meanEscape = Array.from(escapeSamples).reduce((a, b) => a + b, 0) / nSamples;

    // Display
    document.getElementById('pbs-sigma-results').style.display = '';
    document.getElementById('pbs-sigma-chart-wrap').style.display = '';

    document.getElementById('pbs-sigma-median').textContent = median.toFixed(2);
    document.getElementById('pbs-sigma-ci').textContent = `[${ci025.toFixed(2)}, ${ci975.toFixed(2)}]`;
    document.getElementById('pbs-sigma-p3').textContent = (pGt3 * 100).toFixed(1) + '%';
    document.getElementById('pbs-sigma-p6').textContent = (pGt6 * 100).toFixed(1) + '%';
    document.getElementById('pbs-sigma-escape').textContent = (meanEscape * 100).toFixed(2) + '%';

    // Probability bars
    const probs = [
        { label: 'P(\u03A3_B > 2)', value: pGt2, color: '#c0392b' },
        { label: 'P(\u03A3_B > 3)', value: pGt3, color: '#e8c547' },
        { label: 'P(\u03A3_B > 4)', value: pGt4, color: '#4a9f6e' },
        { label: 'P(\u03A3_B > 6)', value: pGt6, color: '#3a7f8f' },
    ];
    document.getElementById('pbs-sigma-bars').innerHTML = probs.map(p => `
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
            <span style="min-width:120px; text-align:right; font-size:12px; font-family:var(--mono); color:var(--text-secondary);">${p.label}</span>
            <div style="flex:1; height:22px; background:var(--bg-secondary); border-radius:4px; overflow:hidden;">
                <div style="width:${(p.value*100).toFixed(1)}%; height:100%; background:${p.color}; border-radius:4px; display:flex; align-items:center; justify-content:flex-end; padding-right:6px; font-size:11px; font-weight:600; color:var(--bg-primary); font-family:var(--mono);">${(p.value*100).toFixed(1)}%</div>
            </div>
        </div>
    `).join('');

    // Insight
    let msg = '';
    if (pGt6 > 0.95) {
        msg = `<strong>Excellent escape resistance.</strong> \u03A3<sub>B</sub> median = ${median.toFixed(2)}, P(\u03A3<sub>B</sub> > 6) = ${(pGt6*100).toFixed(0)}%. Process is far from spec limits with minimal momentum toward them.`;
    } else if (pGt3 > 0.90) {
        msg = `<strong>Adequate margin.</strong> \u03A3<sub>B</sub> median = ${median.toFixed(2)}, P(\u03A3<sub>B</sub> > 3) = ${(pGt3*100).toFixed(0)}%. Reasonable distance from spec.`;
    } else if (pGt3 > 0.50) {
        msg = `<strong>Marginal.</strong> \u03A3<sub>B</sub> median = ${median.toFixed(2)}. Only ${(pGt3*100).toFixed(0)}% chance of exceeding 3\u03C3 equivalent. Process may drift to spec.`;
    } else {
        msg = `<strong>High escape risk.</strong> \u03A3<sub>B</sub> median = ${median.toFixed(2)}. Process is close to spec limits or trending toward them.`;
    }
    if (alignmentPlus > 0.1) {
        msg += ` <em>Momentum factor: process trending toward nearest spec (\uD835\uDCCE\u207A = ${alignmentPlus.toFixed(2)}), reducing effective margin.</em>`;
    }
    msg += ` <br><span style="font-size:11px; color:var(--text-dim);">Escape probability (next ${T_escape} obs): ${(meanEscape*100).toFixed(2)}% via Kramers\u2019 law. Distance to nearest spec: ${Math.min(distToUSL, distToLSL).toFixed(3)}, trend slope: ${trend.toFixed(4)}/obs.</span>`;
    document.getElementById('pbs-sigma-insight').innerHTML = msg;

    // Plotly histogram
    const nBins = 60;
    const lo = sorted[Math.floor(sorted.length * 0.005)];
    const hi = sorted[Math.floor(sorted.length * 0.995)];
    const filtered = sorted.filter(v => v >= lo && v <= hi);

    Plotly.newPlot('pbs-sigma-chart', [{
        x: filtered, type: 'histogram', nbinsx: nBins,
        marker: { color: 'rgba(58,127,143,0.5)', line: { color: 'rgba(58,127,143,0.8)', width: 1 } },
        hovertemplate: '\u03A3_B: %{x:.2f}<br>Count: %{y}<extra></extra>',
    }], {
        shapes: [
            { type: 'line', x0: 3, x1: 3, y0: 0, y1: 1, yref: 'paper', line: { color: '#e8c547', width: 2, dash: 'dash' } },
            { type: 'line', x0: 6, x1: 6, y0: 0, y1: 1, yref: 'paper', line: { color: '#4a9f6e', width: 2, dash: 'dash' } },
            { type: 'line', x0: median, x1: median, y0: 0, y1: 1, yref: 'paper', line: { color: '#3a7f8f', width: 2.5 } },
        ],
        annotations: [
            { x: 3, y: 1, yref: 'paper', text: '3\u03C3', showarrow: false, yanchor: 'bottom', font: { size: 10, color: '#e8c547' } },
            { x: 6, y: 1, yref: 'paper', text: '6\u03C3', showarrow: false, yanchor: 'bottom', font: { size: 10, color: '#4a9f6e' } },
            { x: median, y: 1, yref: 'paper', text: 'Median', showarrow: false, yanchor: 'bottom', font: { size: 10, color: '#3a7f8f' } },
        ],
        xaxis: { title: 'Bayesian \u03A3_B', color: '#9aaa9a' },
        yaxis: { title: 'Count', color: '#9aaa9a' },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#9aaa9a' }, margin: { t: 30, r: 20, b: 50, l: 50 },
    }, { responsive: true, displayModeBar: false });

    // Publish
    SvendOps.publish('pbsSigmaB', parseFloat(median.toFixed(2)), '\u03C3', 'Bayesian Sigma');
    SvendOps.publish('pbsEscapeProb', parseFloat((meanEscape*100).toFixed(2)), '%', 'Bayesian Sigma');

    // Next steps
    document.getElementById('pbs-sigma-next-steps').style.display = '';
    renderNextSteps('pbs-sigma-next-steps', [
        { title: 'Bayesian Cpk', desc: 'Full posterior capability analysis', calcId: 'pbs-cpk' },
        { title: 'Belief Chart', desc: 'Detect when shifts occurred', calcId: 'pbs-belief' },
        { title: 'Evidence Strength', desc: 'Sequential evidence accumulation', calcId: 'pbs-evidence' },
    ]);
}


// ============================================================================
// FMEA Monte Carlo — Risk Distribution Simulation
// ============================================================================

// FMEA Monte Carlo — see calc-sim-quality.js

// SMED Simulator — see calc-sim-quality.js

// Heijunka Simulator — see calc-sim-quality.js
