// DOE Workbench — Power Analysis
// Load order: doe-state.js → doe-design.js → doe-analysis.js → doe-optimize.js → doe-power.js → doe-chat.js

// Power Analysis — live curve with cached grid
let powerCurveData = null;  // cached grid from server

function updateEffectSize() {
    const value = parseFloat(document.getElementById('effect-size').value);
    document.getElementById('effect-value').textContent = value.toFixed(2);

    let label = 'Medium';
    if (value < 0.3) label = 'Small';
    else if (value >= 0.8) label = 'Large';
    document.getElementById('effect-label').textContent = label;

    // If we have a cached curve, interpolate and update cursor instantly
    if (powerCurveData && powerCurveData.length > 0) {
        const interp = interpolatePowerCurve(value);
        if (interp) {
            document.getElementById('sample-size-result').textContent = interp.n;
            document.getElementById('per-group-result').textContent =
                interp.n_per_group ? `(${interp.n_per_group} per group)` : '';
            updatePowerCurveCursor(value, interp.n);
        }
    }
}

function interpolatePowerCurve(d) {
    // Find bracketing points and linearly interpolate
    const pts = powerCurveData;
    if (d <= pts[0].d) return pts[0];
    if (d >= pts[pts.length - 1].d) return pts[pts.length - 1];
    for (let i = 0; i < pts.length - 1; i++) {
        if (d >= pts[i].d && d <= pts[i + 1].d) {
            const t = (d - pts[i].d) / (pts[i + 1].d - pts[i].d);
            const n = Math.round(pts[i].n + t * (pts[i + 1].n - pts[i].n));
            let npg = null;
            if (pts[i].n_per_group != null && pts[i + 1].n_per_group != null) {
                npg = Math.round(pts[i].n_per_group + t * (pts[i + 1].n_per_group - pts[i].n_per_group));
            }
            return { n: n, n_per_group: npg };
        }
    }
    return pts[pts.length - 1];
}

function renderPowerCurve(curve, currentD) {
    const chartEl = document.getElementById('power-curve-chart');
    if (!chartEl) return;

    const ds = curve.map(p => p.d);
    const ns = curve.map(p => p.n);

    // Find current point
    const interp = interpolatePowerCurve(currentD);
    const currentN = interp ? interp.n : null;

    const traces = [
        {
            x: ds, y: ns, type: 'scatter', mode: 'lines',
            name: 'Required n',
            line: { color: '#60a5fa', width: 2.5 },
            hovertemplate: 'Effect size: %{x:.2f}<br>Sample size: %{y}<extra></extra>',
        },
    ];

    // Add cursor marker at current effect size
    if (currentN != null) {
        traces.push({
            x: [currentD], y: [currentN], type: 'scatter', mode: 'markers',
            name: 'Current',
            marker: { color: '#f97316', size: 12, symbol: 'circle',
                      line: { color: '#fff', width: 2 } },
            hovertemplate: 'd=%{x:.2f}, n=%{y}<extra></extra>',
        });
    }

    const layout = {
        margin: { t: 10, r: 20, b: 40, l: 55 },
        xaxis: { title: 'Effect Size (d)', range: [0.08, 2.02], dtick: 0.2 },
        yaxis: { title: 'Sample Size (n)', rangemode: 'tozero' },
        showlegend: false,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#e2e8f0', size: 11 },
        shapes: [
            // Reference lines for Small/Medium/Large
            { type: 'line', x0: 0.2, x1: 0.2, y0: 0, y1: 1, yref: 'paper',
              line: { color: 'rgba(148,163,184,0.3)', width: 1, dash: 'dot' } },
            { type: 'line', x0: 0.5, x1: 0.5, y0: 0, y1: 1, yref: 'paper',
              line: { color: 'rgba(148,163,184,0.3)', width: 1, dash: 'dot' } },
            { type: 'line', x0: 0.8, x1: 0.8, y0: 0, y1: 1, yref: 'paper',
              line: { color: 'rgba(148,163,184,0.3)', width: 1, dash: 'dot' } },
        ],
        annotations: [
            { x: 0.2, y: 1.02, yref: 'paper', text: 'S', showarrow: false,
              font: { size: 9, color: 'rgba(148,163,184,0.6)' } },
            { x: 0.5, y: 1.02, yref: 'paper', text: 'M', showarrow: false,
              font: { size: 9, color: 'rgba(148,163,184,0.6)' } },
            { x: 0.8, y: 1.02, yref: 'paper', text: 'L', showarrow: false,
              font: { size: 9, color: 'rgba(148,163,184,0.6)' } },
        ],
    };

    Plotly.react(chartEl, traces, layout, { responsive: true, displayModeBar: false });
}

function updatePowerCurveCursor(d, n) {
    const chartEl = document.getElementById('power-curve-chart');
    if (!chartEl || !chartEl.data || chartEl.data.length < 2) return;

    Plotly.restyle(chartEl, { x: [[d]], y: [[n]] }, [1]);
}

function updatePowerHelp() {
    const testType = document.getElementById('test-type').value;
    document.getElementById('groups-container').style.display =
        testType === 'anova' ? 'block' : 'none';
}

async function runPowerAnalysis() {
    const data = {
        effect_size: parseFloat(document.getElementById('effect-size').value),
        test_type: document.getElementById('test-type').value,
        alpha: parseFloat(document.getElementById('alpha').value),
        power: parseFloat(document.getElementById('power-level').value),
        groups: parseInt(document.getElementById('num-groups').value),
        include_curve: true,
    };

    try {
        const response = await fetch('/api/experimenter/power/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (response.ok) {
            document.getElementById('power-output-empty').style.display = 'none';
            document.getElementById('power-output').style.display = 'block';

            document.getElementById('sample-size-result').textContent = result.power_analysis.total_sample_size;
            document.getElementById('per-group-result').textContent =
                result.power_analysis.sample_size_per_group ?
                `(${result.power_analysis.sample_size_per_group} per group)` : '';

            document.getElementById('power-interpretation').innerHTML = `
                <p>${result.interpretation.summary}</p>
                <p><strong>Effect Size:</strong> ${result.power_analysis.effect_size} (${result.interpretation.effect_size_meaning})</p>
            `;

            // Cache and render power curve
            if (result.power_curve) {
                powerCurveData = result.power_curve;
                renderPowerCurve(powerCurveData, data.effect_size);
            }
        } else {
            alert(result.error || 'Power analysis failed');
        }
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

// Debounced refresh — re-fetches curve grid when alpha/power/test-type change
let _powerRefreshTimer = null;
function debouncedPowerRefresh() {
    if (!powerCurveData) return;  // only refresh if curve was already shown
    clearTimeout(_powerRefreshTimer);
    _powerRefreshTimer = setTimeout(() => runPowerAnalysis(), 300);
}
