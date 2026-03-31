/**
 * VSM Metrics Module
 * Live calculations, bottleneck detection, step detail panel, takt.
 * MIGRATION: Extracted from templates/vsm.html
 */


function calculateTakt() {
    const avail = parseFloat(document.getElementById('takt-avail').value);
    const demand = parseFloat(document.getElementById('takt-demand').value);
    if (!avail || !demand || demand <= 0 || !currentVSM) return;
    saveVSMState();
    const takt = Math.round((avail / demand) * 10) / 10;
    currentVSM.takt_time = takt;
    document.getElementById('takt-direct').value = takt;
    updateMetrics();
    saveVSM();
}


function closeStepMetrics() {
    document.getElementById('step-metrics-panel').classList.remove('visible');
}

// --- Hypothesis-driven kaizen tracking (Phase 5) ---


function detectBottleneckClient(vsm) {
    // Client-side bottleneck detection — mirrors server logic
    const steps = vsm.process_steps || [];
    const wcs = vsm.work_centers || [];
    if (steps.length === 0) return null;

    const wcMap = {};
    wcs.forEach(wc => { wcMap[wc.id] = { ...wc, members: [] }; });
    const standalone = [];
    steps.forEach(s => {
        if (s.work_center_id && wcMap[s.work_center_id]) {
            wcMap[s.work_center_id].members.push(s);
        } else {
            standalone.push(s);
        }
    });

    const effective = [];
    standalone.forEach(s => {
        const ct = s.cycle_time || 0;
        if (ct > 0) effective.push({ name: s.name || 'Process', ct, id: s.id });
    });
    Object.values(wcMap).forEach(wc => {
        if (wc.members.length === 0) return;
        const rateSum = wc.members.reduce((sum, m) => {
            const ct = m.cycle_time || 0;
            return ct > 0 ? sum + 1.0 / ct : sum;
        }, 0);
        if (rateSum > 0) {
            effective.push({ name: wc.name || wc.members[0].name, ct: 1.0 / rateSum, id: wc.members[0].id });
        }
    });

    if (effective.length === 0) return null;
    const maxCT = Math.max(...effective.map(e => e.ct));
    const bn = effective.find(e => e.ct === maxCT);
    return { name: bn.name, ct: maxCT, throughput: 3600.0 / maxCT, id: bn.id };
}


function removeAnnotation(stepId, idx) {
    if (!currentVSM) return;
    const step = (currentVSM.process_steps || []).find(s => s.id === stepId);
    if (!step || !step.annotations) return;
    saveVSMState();
    step.annotations.splice(idx, 1);
    showStepMetrics(step);
    renderVSM();
    saveVSM();
}


function renderAnnotationCard(a, step, idx) {
    const statusClass = a.status || '';
    const source = (a.source || 'unknown').replace(/_/g, ' ');
    const ts = a.timestamp ? new Date(a.timestamp).toLocaleDateString() : '';
    let body = '';
    if (a.value && typeof a.value === 'object') {
        body = Object.entries(a.value).map(([k, v]) =>
            `<span style="margin-right:8px;">${k.replace(/_/g, ' ')}: <strong>${v}</strong></span>`
        ).join('');
    } else if (a.value) {
        body = String(a.value);
    }
    const stepId = step.id || '';
    return `<div class="smp-annotation-card ${statusClass}">
        <button class="smp-ann-remove" onclick="removeAnnotation('${stepId}', ${idx})" title="Remove">&times;</button>
        <div class="smp-ann-source">${source}</div>
        <div class="smp-ann-body">${body}</div>
        ${ts ? '<div class="smp-ann-time">' + ts + '</div>' : ''}
    </div>`;
}


function renderSuggestedCalcs(bottleneck) {
    const container = document.getElementById('suggested-calcs');
    const list = document.getElementById('suggested-calcs-list');
    const suggestions = [];

    if (!currentVSM || (currentVSM.process_steps || []).length === 0) {
        container.style.display = 'none';
        return;
    }

    if (!currentVSM.takt_time) {
        suggestions.push({ name: 'Takt Time', id: 'takt', reason: 'Set takt to flag overloaded steps' });
    }
    if (bottleneck) {
        if (currentVSM.takt_time && bottleneck.ct > currentVSM.takt_time) {
            suggestions.push({ name: 'SMED', id: 'smed', reason: 'Reduce changeover at bottleneck' });
            suggestions.push({ name: 'Line Simulator', id: 'line-sim', reason: 'Model flow impact' });
        } else {
            suggestions.push({ name: 'Bottleneck Analysis', id: 'bottleneck', reason: 'Detailed constraint analysis' });
        }
    }
    if ((currentVSM.process_steps || []).some(s => !s.uptime || s.uptime < 100)) {
        suggestions.push({ name: 'OEE', id: 'oee', reason: 'Measure equipment effectiveness' });
    }

    if (suggestions.length === 0) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'block';
    list.innerHTML = suggestions.slice(0, 3).map(s =>
        `<a href="/app/calculators/" target="_blank" style="display:block; padding:0.3rem 0.4rem; margin-bottom:0.25rem; background:var(--bg-tertiary); border-radius:4px; text-decoration:none; font-size:0.75rem; color:var(--accent-primary);" title="${s.reason}">
            ${s.name} <span style="color:var(--text-dim); font-size:0.65rem;">— ${s.reason}</span>
        </a>`
    ).join('');
}

// --- Step Metrics Overlay ---
let stepMetricsClickTimer = null;


function setTaktDirect() {
    const val = parseFloat(document.getElementById('takt-direct').value);
    if (!val || val <= 0 || !currentVSM) return;
    saveVSMState();
    currentVSM.takt_time = Math.round(val * 10) / 10;
    updateMetrics();
    saveVSM();
}


function showStepMetrics(step) {
    const panel = document.getElementById('step-metrics-panel');
    document.getElementById('smp-title').textContent = step.name || 'Unnamed Step';

    // Banner: bottleneck / exceeds takt / normal
    const flags = step.flags || {};
    const bannerEl = document.getElementById('smp-banner');
    if (flags.is_bottleneck) {
        bannerEl.innerHTML = '<div class="smp-banner red">System Constraint (Bottleneck)</div>';
    } else if (flags.exceeds_takt) {
        bannerEl.innerHTML = '<div class="smp-banner yellow">Exceeds Takt — ratio ' + (flags.takt_ratio || '?') + '</div>';
    } else if (flags.takt_ratio) {
        bannerEl.innerHTML = '<div class="smp-banner green">Within Takt — ratio ' + flags.takt_ratio + '</div>';
    } else {
        bannerEl.innerHTML = '';
    }

    // KPI grid
    const ct = step.cycle_time || '-';
    const co = step.changeover_time || '-';
    const uptime = step.uptime ? step.uptime + '%' : '-';
    const ops = step.operators || '-';
    const taktRatio = flags.takt_ratio || '-';
    const scrap = step.scrap_rate ? step.scrap_rate + '%' : '-';
    document.getElementById('smp-kpis').innerHTML =
        `<div class="smp-kpi"><div class="smp-kpi-label">C/T (sec)</div><div class="smp-kpi-value">${ct}</div></div>` +
        `<div class="smp-kpi"><div class="smp-kpi-label">C/O (sec)</div><div class="smp-kpi-value">${co}</div></div>` +
        `<div class="smp-kpi"><div class="smp-kpi-label">Uptime</div><div class="smp-kpi-value">${uptime}</div></div>` +
        `<div class="smp-kpi"><div class="smp-kpi-label">Operators</div><div class="smp-kpi-value">${ops}</div></div>` +
        `<div class="smp-kpi"><div class="smp-kpi-label">vs Takt</div><div class="smp-kpi-value">${taktRatio}</div></div>` +
        `<div class="smp-kpi"><div class="smp-kpi-label">Scrap</div><div class="smp-kpi-value">${scrap}</div></div>`;

    // Annotations
    const annotations = step.annotations || [];
    const annEl = document.getElementById('smp-annotations');
    if (annotations.length > 0) {
        annEl.innerHTML = '<div class="smp-section-title">Linked Results</div>' +
            annotations.map((a, i) => renderAnnotationCard(a, step, i)).join('');
    } else {
        annEl.innerHTML = '<div class="smp-section-title">No linked results</div>' +
            '<div style="font-size:0.75rem; color:var(--text-dim); padding:4px 0;">Run a calculator and export to VSM to see results here.</div>';
    }

    panel.classList.add('visible');
}

// =============================================================================
function updateMetrics() {
    if (!currentVSM) return;

    // Calculate totals (work center aware — parallel machines use effective CT)
    let totalCT = 0;
    let totalWait = 0;

    const wcSteps = {};
    (currentVSM.process_steps || []).forEach(step => {
        const ct = step.cycle_time || 0;
        if (step.work_center_id) {
            if (!wcSteps[step.work_center_id]) wcSteps[step.work_center_id] = [];
            wcSteps[step.work_center_id].push(ct);
        } else {
            totalCT += ct;
        }
    });
    // Add effective CT for each work center
    Object.values(wcSteps).forEach(cts => {
        const rateSum = cts.reduce((s, ct) => ct > 0 ? s + 1.0/ct : s, 0);
        if (rateSum > 0) totalCT += 1.0 / rateSum;
    });

    (currentVSM.inventory || []).forEach(inv => {
        totalWait += inv.days_of_supply || 0;
    });

    const leadTime = totalWait + (totalCT / 86400);
    const pce = leadTime > 0 ? ((totalCT / 86400) / leadTime * 100) : 0;

    document.getElementById('metric-lead-time').textContent = leadTime.toFixed(2) + ' days';
    document.getElementById('metric-process-time').textContent = totalCT + ' sec';
    document.getElementById('metric-pce').textContent = pce.toFixed(1) + '%';
    document.getElementById('metric-takt').textContent = currentVSM.takt_time ? currentVSM.takt_time + ' sec' : '-';

    // Pre-fill takt inputs if data exists
    if (currentVSM.takt_time) {
        document.getElementById('takt-direct').value = currentVSM.takt_time;
    }

    // Client-side bottleneck detection
    const bn = detectBottleneckClient(currentVSM);
    const bnEl = document.getElementById('metric-bottleneck');
    const tpEl = document.getElementById('metric-throughput');
    if (bn) {
        bnEl.textContent = bn.name + ' (' + formatTime(bn.ct) + ')';
        bnEl.style.color = (currentVSM.takt_time && bn.ct > currentVSM.takt_time) ? '#e74c3c' : '';
        tpEl.textContent = bn.throughput.toFixed(1) + ' u/hr';
    } else {
        bnEl.textContent = '-';
        bnEl.style.color = '';
        tpEl.textContent = '-';
    }

    // Suggested calculators
    renderSuggestedCalcs(bn);
}

