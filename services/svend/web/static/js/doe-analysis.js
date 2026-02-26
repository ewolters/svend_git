// DOE Workbench — Results Entry, ANOVA Display, Effects Plots, Residuals
// Load order: doe-state.js → doe-design.js → doe-analysis.js → doe-optimize.js → doe-power.js → doe-chat.js

// Interactive alpha — cached analysis data for live updates
let cachedCoefficients = null;  // from analysis response
let cachedParetoData = null;    // sorted pareto bars
let cachedResidualDF = 0;       // residual degrees of freedom

// Analyze Results
async function analyzeResults() {
    if (!currentDesign) {
        alert('Generate a design first');
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

    if (results.length === 0) {
        alert('Please enter at least one response value.');
        return;
    }

    const data = {
        design: currentDesign,
        results,
        response_name: document.getElementById('response-name').value,
        alpha: 0.05,
        include_interactions: true,
        fit_quadratic: ['ccd', 'box_behnken', 'definitive_screening'].includes(currentDesign.design_type?.toLowerCase()),
        problem_id: document.getElementById('project-selector').value || undefined,
    };

    try {
        const response = await fetch('/api/experimenter/analyze/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (response.ok) {
            currentAnalysis = result;
            saveState();
            displayAnalysis(result);
            updateContextBadge();
            // Wizard: mark results complete and advance to analyze step
            completeStep('design');
            completeStep('results');
            completeStep('analyze');
            goToStep('analyze');
            showSubTab('step-analyze', 'anova');
        } else {
            alert(result.error || 'Analysis failed');
        }
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

function displayAnalysis(result) {
    // Show ANOVA content
    document.getElementById('no-analysis-message').style.display = 'none';
    document.getElementById('anova-content').style.display = 'block';

    const analysis = result.analysis;

    // Cache coefficients + DF for interactive alpha slider
    cachedCoefficients = analysis.coefficients || [];
    // Extract residual DF from ANOVA table (second row = Residual Error)
    const anovaDF = analysis.anova_table?.df;
    cachedResidualDF = (anovaDF && anovaDF.length >= 2) ? anovaDF[1] : 0;
    // Reset alpha slider to 0.05
    const alphaSlider = document.getElementById('alpha-effects-slider');
    if (alphaSlider) alphaSlider.value = '0.05';
    const alphaLabel = document.getElementById('alpha-effects-value');
    if (alphaLabel) alphaLabel.textContent = '0.05';

    // Model summary — with R² color-coding
    const r2El = document.getElementById('r-squared');
    const r2Val = parseFloat(analysis.model_summary.r_squared);
    r2El.textContent = analysis.model_summary.r_squared + '%';
    r2El.className = 'stat-value ' + (r2Val >= 90 ? 'r2-excellent' : r2Val >= 70 ? 'r2-good' : r2Val >= 50 ? 'r2-moderate' : 'r2-poor');

    const r2AdjEl = document.getElementById('r-squared-adj');
    const r2AdjVal = parseFloat(analysis.model_summary.r_squared_adj);
    r2AdjEl.textContent = analysis.model_summary.r_squared_adj != null ? analysis.model_summary.r_squared_adj + '%' : 'N/A';
    if (!isNaN(r2AdjVal)) {
        r2AdjEl.className = 'stat-value ' + (r2AdjVal >= 90 ? 'r2-excellent' : r2AdjVal >= 70 ? 'r2-good' : r2AdjVal >= 50 ? 'r2-moderate' : 'r2-poor');
    } else {
        r2AdjEl.className = 'stat-value';
    }

    document.getElementById('model-s').textContent = analysis.model_summary.s || 'N/A';

    // Saturated model warning
    const satWarn = document.getElementById('saturated-warning');
    if (satWarn) satWarn.remove();
    if (analysis.model_summary.saturated) {
        const warn = document.createElement('div');
        warn.id = 'saturated-warning';
        warn.style.cssText = 'background: rgba(255,180,0,0.1); border: 1px solid rgba(255,180,0,0.3); border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.85rem; color: var(--text-secondary);';
        warn.innerHTML = '<strong>Saturated model</strong> — all degrees of freedom are used to estimate effects. P-values are not available. Add replicates or center points to enable significance testing.';
        document.getElementById('anova-content').prepend(warn);
    }

    // ANOVA table
    const anova = analysis.anova_table;
    let anovaHtml = `
        <table class="data-table">
            <thead>
                <tr><th>Source</th><th class="right">DF</th><th class="right">SS</th><th class="right">MS</th><th class="right">F</th><th class="right">P</th></tr>
            </thead>
            <tbody>
    `;
    const alpha = parseFloat(document.getElementById('alpha-effects-slider')?.value || '0.05');
    for (let i = 0; i < anova.source.length; i++) {
        let pCell = '-';
        if (anova.p[i] != null) {
            const pVal = anova.p[i];
            const isSig = pVal < alpha;
            const isWarn = !isSig && pVal < alpha * 2;
            const badgeClass = isSig ? 'sig-yes' : isWarn ? 'sig-warn' : 'sig-no';
            const badgeLabel = isSig ? 'Sig' : isWarn ? 'Marginal' : 'NS';
            pCell = `${pVal} <span class="sig-badge ${badgeClass}">${badgeLabel}</span>`;
        }
        const rowClass = anova.p[i] != null && anova.p[i] < alpha ? ' class="anova-significant-row"' : '';
        anovaHtml += `
            <tr${rowClass}>
                <td>${anova.source[i]}</td>
                <td class="right">${anova.df[i]}</td>
                <td class="right">${anova.ss[i]}</td>
                <td class="right">${anova.ms[i] ?? '-'}</td>
                <td class="right">${anova.f[i] ?? '-'}</td>
                <td class="right">${pCell}</td>
            </tr>
        `;
    }
    anovaHtml += '</tbody></table>';
    document.getElementById('anova-table-container').innerHTML = anovaHtml;

    // Coefficients table
    let coefHtml = `
        <table class="data-table">
            <thead>
                <tr><th>Term</th><th class="right">Effect</th><th class="right">Coef</th><th class="right">SE</th><th class="right">T</th><th class="right">P</th></tr>
            </thead>
            <tbody>
    `;
    analysis.coefficients.forEach(c => {
        const sigClass = c.significant === true ? 'significant' : (c.significant === false ? 'not-significant' : '');
        coefHtml += `
            <tr>
                <td>${c.term}</td>
                <td class="right">${c.effect ?? '-'}</td>
                <td class="right">${c.coefficient}</td>
                <td class="right">${c.se_coef ?? '-'}</td>
                <td class="right">${c.t_value ?? '-'}</td>
                <td class="right ${sigClass}">${c.p_value ?? '-'}</td>
            </tr>
        `;
    });
    coefHtml += '</tbody></table>';
    document.getElementById('coefficients-table-container').innerHTML = coefHtml;

    // Model equation
    document.getElementById('model-equation').textContent = analysis.model_equation;

    // Interpretation
    document.getElementById('interpretation-box').innerHTML =
        result.interpretation.map(i => `<p>${i}</p>`).join('');

    // Update effects plots
    displayEffectsPlots(result.plots);

    // Update residuals
    displayResiduals(result);

    // Update contour panel
    updateContourPanel();

    // Update optimizer
    updateOptimizerPanel();
}

function displayEffectsPlots(plots) {
    document.getElementById('no-effects-message').style.display = 'none';
    document.getElementById('effects-content').style.display = 'block';

    // Main Effects Plot
    if (plots.main_effects?.length > 0) {
        const traces = plots.main_effects.map(me => ({
            x: me.levels.map(String),
            y: me.means,
            type: 'scatter',
            mode: 'lines+markers',
            name: me.factor,
            line: { width: 2 },
            marker: { size: 8 },
        }));

        Plotly.newPlot('main-effects-plot', traces, {
            title: 'Main Effects Plot',
            xaxis: { title: 'Factor Level' },
            yaxis: { title: 'Mean Response' },
            showlegend: true,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') },
        }, { responsive: true });
    }

    // Interaction Plots
    if (plots.interactions?.length > 0) {
        const interactionTraces = [];
        plots.interactions.forEach((ip, idx) => {
            ip.data.forEach(series => {
                interactionTraces.push({
                    x: series.points.map(p => String(p.x)),
                    y: series.points.map(p => p.y),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: `${ip.factors[0]}=${series.level}`,
                });
            });
        });

        Plotly.newPlot('interactions-plot', interactionTraces, {
            title: 'Interaction Plot',
            xaxis: { title: plots.interactions[0]?.factors[1] || 'Factor 2' },
            yaxis: { title: 'Mean Response' },
            showlegend: true,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') },
        }, { responsive: true });
    }

    // Pareto Chart
    if (plots.pareto?.length > 0) {
        const sorted = plots.pareto.slice().sort((a, b) => b.standardized_effect - a.standardized_effect);
        // Cache pareto data with t-values for alpha slider
        cachedParetoData = sorted.map(p => {
            // Match t_value from cached coefficients
            const coef = cachedCoefficients.find(c => c.term === p.term);
            return { term: p.term, standardized_effect: p.standardized_effect, t_value: coef?.t_value ?? null };
        });

        const colors = sorted.map(p => p.significant ? '#22c55e' : '#6b7280');

        Plotly.newPlot('pareto-plot', [{
            y: sorted.map(p => p.term),
            x: sorted.map(p => p.standardized_effect),
            type: 'bar',
            orientation: 'h',
            marker: { color: colors },
        }], {
            title: 'Pareto Chart of Standardized Effects',
            xaxis: { title: '|Standardized Effect|' },
            shapes: [{
                type: 'line',
                x0: plots.pareto_reference,
                x1: plots.pareto_reference,
                y0: -0.5,
                y1: sorted.length - 0.5,
                line: { color: 'red', dash: 'dash', width: 2 },
            }],
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') },
            margin: { l: 150 },
        }, { responsive: true });
    }
}

function showPlot(plotId) {
    document.querySelectorAll('.plot-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.plot-container').forEach(p => p.classList.remove('active'));

    event.target.classList.add('active');
    document.getElementById(`${plotId}-plot`).classList.add('active');

    window.dispatchEvent(new Event('resize'));
}

// t-distribution inverse (Abramowitz & Stegun 26.2.16 approximation)
// Two-tailed: returns t such that P(|T| > t) = alpha
function tInv2(alpha, df) {
    // Normal quantile approximation (Abramowitz & Stegun 26.2.23)
    const p = alpha / 2;
    const t0 = Math.sqrt(-2 * Math.log(p));
    const z = t0 - (2.515517 + 0.802853 * t0 + 0.010328 * t0 * t0) /
                    (1 + 1.432788 * t0 + 0.189269 * t0 * t0 + 0.001308 * t0 * t0 * t0);
    // Cornish-Fisher correction for t distribution
    const g1 = (z * z * z + z) / (4 * df);
    const g2 = (5 * z * z * z * z * z + 16 * z * z * z + 3 * z) / (96 * df * df);
    return z + g1 + g2;
}

function onAlphaSliderChange() {
    const alpha = parseFloat(document.getElementById('alpha-effects-slider').value);
    document.getElementById('alpha-effects-value').textContent = alpha.toFixed(2);

    if (!cachedCoefficients || cachedResidualDF <= 0) return;

    const tCrit = tInv2(alpha, cachedResidualDF);

    // Update ANOVA table badges
    const anovaRows = document.querySelectorAll('#anova-table-container tbody tr');
    anovaRows.forEach(row => {
        const pCell = row.cells[5];
        if (!pCell) return;
        // Extract raw p-value from text (before badge span)
        const rawText = pCell.textContent.replace(/Sig|Marginal|NS/g, '').trim();
        const pVal = parseFloat(rawText);
        if (isNaN(pVal)) return;
        const isSig = pVal < alpha;
        const isWarn = !isSig && pVal < alpha * 2;
        const badgeClass = isSig ? 'sig-yes' : isWarn ? 'sig-warn' : 'sig-no';
        const badgeLabel = isSig ? 'Sig' : isWarn ? 'Marginal' : 'NS';
        pCell.innerHTML = `${rawText} <span class="sig-badge ${badgeClass}">${badgeLabel}</span>`;
        row.className = isSig ? 'anova-significant-row' : '';
    });

    // Update coefficients table significance flags
    const rows = document.querySelectorAll('#coefficients-table-container tbody tr');
    rows.forEach((row, i) => {
        const coef = cachedCoefficients[i];
        if (!coef || coef.p_value == null) return;
        const pCell = row.cells[5];
        const isSig = coef.p_value < alpha;
        pCell.className = 'right ' + (isSig ? 'significant' : 'not-significant');
    });

    // Update Pareto reference line + bar colors
    if (cachedParetoData && cachedParetoData.length > 0) {
        const paretoEl = document.getElementById('pareto-plot');
        if (paretoEl && paretoEl.data) {
            // Update bar colors based on new t_critical
            const newColors = cachedParetoData.map(p =>
                (p.t_value != null && Math.abs(p.t_value) > tCrit) ? '#22c55e' : '#6b7280'
            );
            Plotly.restyle(paretoEl, { 'marker.color': [newColors] }, [0]);
            Plotly.relayout(paretoEl, {
                'shapes[0].x0': tCrit,
                'shapes[0].x1': tCrit,
            });
        }
    }
}

function displayResiduals(result) {
    document.getElementById('no-residuals-message').style.display = 'none';
    document.getElementById('residuals-content').style.display = 'block';

    const plots = result.plots;
    const diagnostics = result.diagnostics;

    // Normal probability plot
    if (plots.normal_probability) {
        Plotly.newPlot('normal-plot', [{
            x: plots.normal_probability.theoretical,
            y: plots.normal_probability.residuals,
            type: 'scatter',
            mode: 'markers',
            marker: { size: 6, color: '#3b82f6' },
        }], {
            xaxis: { title: 'Theoretical Quantiles' },
            yaxis: { title: 'Residuals' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') },
            margin: { t: 20 },
        }, { responsive: true });

        const ad = plots.normal_probability.anderson_darling;
        if (ad) {
            const status = ad.normal ? 'Normal' : 'Not Normal';
            document.getElementById('normality-test').innerHTML =
                `Anderson-Darling: ${ad.statistic} (critical: ${ad.critical_5pct}) - ${status}`;
        }
    }

    // Residuals vs Fitted
    if (plots.residual_vs_fitted) {
        Plotly.newPlot('fitted-plot', [{
            x: plots.residual_vs_fitted.fitted,
            y: plots.residual_vs_fitted.residuals,
            type: 'scatter',
            mode: 'markers',
            marker: { size: 6, color: '#3b82f6' },
        }], {
            xaxis: { title: 'Fitted Values' },
            yaxis: { title: 'Residuals' },
            shapes: [{ type: 'line', x0: Math.min(...plots.residual_vs_fitted.fitted),
                      x1: Math.max(...plots.residual_vs_fitted.fitted), y0: 0, y1: 0,
                      line: { color: 'gray', dash: 'dash' } }],
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') },
            margin: { t: 20 },
        }, { responsive: true });
    }

    // Residuals vs Order
    if (plots.residual_vs_order) {
        Plotly.newPlot('order-plot', [{
            x: plots.residual_vs_order.order,
            y: plots.residual_vs_order.residuals,
            type: 'scatter',
            mode: 'lines+markers',
            marker: { size: 6, color: '#3b82f6' },
        }], {
            xaxis: { title: 'Run Order' },
            yaxis: { title: 'Residuals' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary') },
            margin: { t: 20 },
        }, { responsive: true });

        const dw = diagnostics.durbin_watson;
        const dwStatus = dw > 1.5 && dw < 2.5 ? 'No autocorrelation' : 'Possible autocorrelation';
        document.getElementById('dw-test').innerHTML = `Durbin-Watson: ${dw} - ${dwStatus}`;
    }

    // Lack of Fit — traffic-light card
    const lof = diagnostics.lack_of_fit;
    if (lof) {
        const cardClass = lof.significant ? 'lof-fail' : 'lof-pass';
        const statusText = lof.significant ? 'Significant Lack of Fit' : 'Model Adequate';
        document.getElementById('lof-result').innerHTML = `
            <div class="lof-card ${cardClass}">
                <div class="lof-status">${statusText}</div>
                <div class="lof-detail">F = ${lof.f_value}, p = ${lof.p_value}</div>
                <div class="lof-interpretation">${lof.interpretation}</div>
            </div>
        `;
    } else {
        document.getElementById('lof-result').innerHTML = `
            <div class="lof-card lof-na">
                <div class="lof-status">No Replicates</div>
                <div class="lof-interpretation">Add replicates or center points to enable lack-of-fit testing.</div>
            </div>
        `;
    }
}

function importResults() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.csv';
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const text = await file.text();
        const lines = text.trim().split('\n');

        if (lines.length < 2) {
            alert('CSV must have a header row and data');
            return;
        }

        const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
        const responseIdx = headers.findIndex(h => h === 'response' || h === 'y' || h === 'result');
        const runIdx = headers.findIndex(h => h === 'run' || h === 'run_order' || h === 'order');

        if (responseIdx === -1) {
            alert('CSV must have a "Response" column');
            return;
        }

        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',');
            const response = parseFloat(values[responseIdx]);
            const runOrder = runIdx >= 0 ? parseInt(values[runIdx]) : i;

            if (!isNaN(response)) {
                const run = currentDesign.runs.find(r => r.run_order === runOrder);
                if (run) {
                    const input = document.querySelector(`.response-input[data-run-id="${run.run_id}"]`);
                    if (input) input.value = response;
                }
            }
        }
    };
    input.click();
}
