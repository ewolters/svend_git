/**
 * calc-vsm.js — VSM Integration for Operations Workbench
 *
 * Load order: after calc-core.js (uses SvendOps, showToast, navigateToCalc)
 * Extracted from: calculators.html (inline script)
 *
 * Provides: VSM import (16 calculator-specific loaders), VSM export
 * (10 calculator-specific exporters), DSW/SPC result pinning.
 */

// ============================================================================
// VSM Import/Export Integration
// ============================================================================

let vsmImportCache = []; // cached list of user's VSMs

async function openVSMImport() {
    const modal = document.getElementById('vsm-import-modal');
    modal.style.display = 'flex';
    const select = document.getElementById('vsm-import-select');
    select.innerHTML = '<option value="">Loading...</option>';
    document.getElementById('vsm-import-preview').style.display = 'none';

    try {
        const resp = await fetch('/api/vsm/', { credentials: 'same-origin' });
        if (!resp.ok) throw new Error('Failed to fetch VSMs');
        const data = await resp.json();
        vsmImportCache = data.maps || [];

        if (vsmImportCache.length === 0) {
            select.innerHTML = '<option value="">No VSMs found — create one in Visual > VSM first</option>';
            return;
        }

        select.innerHTML = '<option value="">— Select a VSM —</option>' +
            vsmImportCache.map((m, i) => `<option value="${i}">${m.name} (${m.process_steps?.length || 0} steps)</option>`).join('');
    } catch (e) {
        select.innerHTML = '<option value="">Error loading VSMs</option>';
    }
}

function closeVSMImport() {
    document.getElementById('vsm-import-modal').style.display = 'none';
}

function previewVSMImport() {
    const idx = document.getElementById('vsm-import-select').value;
    const preview = document.getElementById('vsm-import-preview');
    if (idx === '') { preview.style.display = 'none'; return; }

    const vsm = vsmImportCache[parseInt(idx)];
    if (!vsm) return;
    preview.style.display = 'block';

    const steps = vsm.process_steps || [];
    const wcs = vsm.work_centers || [];
    const effectiveStations = buildEffectiveStations(steps, wcs);
    const stepsEl = document.getElementById('vsm-import-steps');
    if (effectiveStations.length === 0) {
        stepsEl.innerHTML = '<div style="color:var(--text-dim);">No process steps in this VSM</div>';
    } else {
        stepsEl.innerHTML = `<table style="width:100%; border-collapse:collapse;">
            <tr style="color:var(--text-dim); font-size:10px; text-transform:uppercase; letter-spacing:0.5px;">
                <th style="text-align:left; padding:4px 8px;">Station</th>
                <th style="text-align:right; padding:4px 8px;">C/T (s)</th>
                <th style="text-align:right; padding:4px 8px;">Type</th>
            </tr>
            ${effectiveStations.map(s => `<tr style="border-top:1px solid var(--border);">
                <td style="padding:6px 8px; color:var(--text-primary);">${s.name}${s.is_work_center ? ' <span style="color:var(--accent);font-size:10px;">[WC]</span>' : ''}</td>
                <td style="padding:6px 8px; text-align:right; color:var(--text-secondary);">${s.cycle_time ? s.cycle_time.toFixed(1) : '—'}${s.is_work_center ? ' (eff.)' : ''}</td>
                <td style="padding:6px 8px; text-align:right; color:var(--text-dim); font-size:11px;">${s.is_work_center ? s.n_machines + ' machines' : 'Single'}</td>
            </tr>`).join('')}
        </table>`;
    }

    const metaEl = document.getElementById('vsm-import-meta');
    const parts = [];
    if (vsm.takt_time) parts.push(`Takt: ${vsm.takt_time}s`);
    if (vsm.customer_demand) parts.push(`Demand: ${vsm.customer_demand}`);
    if (vsm.product_family) parts.push(`Family: ${vsm.product_family}`);
    if (wcs.length > 0) parts.push(`${wcs.length} work center${wcs.length !== 1 ? 's' : ''}`);
    metaEl.textContent = parts.join(' · ') || 'No additional metadata';
}

function buildEffectiveStations(steps, workCenters) {
    // Collapses work center members into single effective stations.
    // Standalone steps pass through unchanged.
    // Work center effective CT = 1 / sum(1/CT_i) for parallel machines.
    const wcMap = {};
    workCenters.forEach(wc => { wcMap[wc.id] = { ...wc, members: [] }; });

    const standalone = [];
    steps.forEach(s => {
        if (s.work_center_id && wcMap[s.work_center_id]) {
            wcMap[s.work_center_id].members.push(s);
        } else {
            standalone.push(s);
        }
    });

    // Build ordered list by x-position
    const all = [];

    // Add standalone steps
    standalone.forEach(s => {
        all.push({ name: s.name || 'Process', cycle_time: s.cycle_time || 30, x: s.x || 0 });
    });

    // Add work centers as collapsed stations
    Object.values(wcMap).forEach(wc => {
        if (wc.members.length === 0) return;
        const rateSum = wc.members.reduce((sum, m) => {
            const ct = m.cycle_time || 0;
            return ct > 0 ? sum + 1.0 / ct : sum;
        }, 0);
        const effCT = rateSum > 0 ? 1.0 / rateSum : 30;
        const memberNames = wc.members.map(m => m.name).join('+');
        all.push({
            name: wc.name || memberNames,
            cycle_time: Math.round(effCT * 10) / 10,
            x: wc.x || 0,
            is_work_center: true,
            n_machines: wc.members.length
        });
    });

    // Sort by x position to maintain process order
    all.sort((a, b) => a.x - b.x);
    return all;
}

function doVSMImport() {
    const idx = document.getElementById('vsm-import-select').value;
    if (idx === '') return;
    const vsm = vsmImportCache[parseInt(idx)];
    if (!vsm) return;

    const steps = vsm.process_steps || [];
    const wcs = vsm.work_centers || [];
    const id = currentCalcId;

    // Build effective station list: standalone steps + work centers (collapsed)
    const effectiveStations = buildEffectiveStations(steps, wcs);

    // Dispatch to calculator-specific import functions
    switch (id) {
        case 'line-sim': loadVSMIntoLineSim(effectiveStations); break;
        case 'kanban-sim': loadVSMIntoKanbanSim(effectiveStations); break;
        case 'toc-sim': loadVSMIntoTocSim(effectiveStations); break;
        case 'bottleneck': loadVSMIntoBottleneck(effectiveStations); break;
        case 'yamazumi': loadYamazumiFromVSMExpanded(steps, wcs); break;
        case 'cell-sim': loadVSMIntoCellSim(effectiveStations); break;
        case 'safety-sim': loadVSMIntoSafetySim(effectiveStations); break;
        case 'takt': loadVSMIntoTakt(vsm); break;
        case 'oee': loadVSMIntoOEE(steps); break;
        case 'kanban': loadVSMIntoKanbanSizing(vsm); break;
        case 'capacity-load': loadVSMIntoCapacityLoad(effectiveStations, vsm); break;
        case 'rto': loadVSMIntoRTO(effectiveStations, vsm); break;
        default:
            // Generic: try to load into simulators that have stations
            if (effectiveStations.length > 0) {
                loadVSMIntoLineSim(effectiveStations);
                alert(`Imported ${effectiveStations.length} stations. Switched to Line Simulator.`);
                document.querySelector('[onclick="showCalc(\'line-sim\')"]')?.click();
                closeVSMImport();
                return;
            }
            alert('This calculator does not support VSM import directly.');
            return;
    }

    closeVSMImport();
}

// --- Calculator-specific VSM import functions ---

function loadVSMIntoLineSim(stations) {
    if (stations.length === 0) return;
    lineStations.length = 0;
    stations.forEach(s => {
        lineStations.push({ name: s.name, cycleTime: s.cycle_time });
    });
    renderLineStations();
    resetLineSim();
}

function loadVSMIntoKanbanSim(stations) {
    if (stations.length === 0) return;
    kanbanStations.length = 0;
    stations.forEach(s => {
        kanbanStations.push({ name: s.name, cycleTime: s.cycle_time });
    });
    renderKanbanStations();
    resetKanbanSim();
}

function loadVSMIntoTocSim(stations) {
    if (stations.length === 0) return;
    tocStationsData.length = 0;
    stations.forEach(s => {
        const ct = s.cycle_time || 60;
        // Work centers with N machines have combined capacity = N * (3600/effective_CT) isn't right
        // effective_CT already accounts for parallel machines, so capacity = 3600 / effective_CT
        const capacity = Math.round(3600 / ct);
        tocStationsData.push({ name: s.name, capacity });
    });
    renderTocStations();
    resetTocSim();
}

function loadVSMIntoBottleneck(stations) {
    if (stations.length === 0) return;
    bottleneckData.length = 0;
    stations.forEach(s => {
        bottleneckData.push({ name: s.name, time: s.cycle_time });
    });
    renderBottleneckStations();
    calcBottleneck();
}

// Yamazumi stores raw VSM import data for toggle between expanded/collapsed work centers
let yamazumiVSMRaw = null; // { steps: [...], workCenters: [...] }

function loadVSMIntoYamazumi(stations) {
    if (stations.length === 0) return;
    yamazumiData.length = 0;
    stations.forEach(s => {
        yamazumiData.push({ name: s.name, time: s.cycle_time });
    });
    renderYamazumiInputs();
    renderYamazumi();
}

function loadYamazumiFromVSMExpanded(steps, workCenters) {
    // Store raw data for toggle
    yamazumiVSMRaw = { steps: JSON.parse(JSON.stringify(steps)), workCenters: JSON.parse(JSON.stringify(workCenters)) };

    // Show toggle only when work centers exist
    const toggle = document.getElementById('yama-wc-toggle');
    const hasWC = workCenters.length > 0 && steps.some(s => s.work_center_id);
    toggle.style.display = hasWC ? 'flex' : 'none';

    // Reset checkbox to unchecked (expanded = default)
    document.getElementById('yama-combine-wc').checked = false;

    // Expanded: each machine is its own bar, sorted by x position
    yamazumiData.length = 0;
    [...steps].sort((a, b) => (a.x || 0) - (b.x || 0)).forEach(s => {
        const wcId = s.work_center_id;
        let prefix = '';
        if (wcId) {
            const wc = workCenters.find(w => w.id === wcId);
            if (wc) prefix = wc.name + ': ';
        }
        yamazumiData.push({ name: prefix + (s.name || 'Process'), time: s.cycle_time || 30 });
    });
    renderYamazumiInputs();
    renderYamazumi();
}

function toggleYamazumiWCMode() {
    if (!yamazumiVSMRaw) return;
    const combine = document.getElementById('yama-combine-wc').checked;

    const { steps, workCenters } = yamazumiVSMRaw;

    if (combine) {
        // Collapsed: use buildEffectiveStations
        const effective = buildEffectiveStations(steps, workCenters);
        yamazumiData.length = 0;
        effective.forEach(s => {
            yamazumiData.push({
                name: s.is_work_center ? s.name + ' (eff.)' : s.name,
                time: s.cycle_time
            });
        });
    } else {
        // Expanded: each machine
        yamazumiData.length = 0;
        [...steps].sort((a, b) => (a.x || 0) - (b.x || 0)).forEach(s => {
            const wcId = s.work_center_id;
            let prefix = '';
            if (wcId) {
                const wc = workCenters.find(w => w.id === wcId);
                if (wc) prefix = wc.name + ': ';
            }
            yamazumiData.push({ name: prefix + (s.name || 'Process'), time: s.cycle_time || 30 });
        });
    }
    renderYamazumiInputs();
    renderYamazumi();
}

function loadVSMIntoTakt(vsm) {
    if (vsm.takt_time) {
        const taktSec = vsm.takt_time;
        // Reverse-engineer: takt = available / demand, so if we know takt we can set inputs
        // Just set available=480 and calculate demand from takt
        const available = parseFloat(document.getElementById('takt-available').value) || 480;
        const breaks = parseFloat(document.getElementById('takt-breaks').value) || 0;
        const net = available - breaks;
        const demand = Math.round(net / (taktSec / 60)); // takt is in sec, available is in min
        document.getElementById('takt-demand').value = demand;
        calcTakt();
    }
    if (vsm.customer_demand) {
        // Try to parse a number from customer_demand string
        const num = parseFloat(vsm.customer_demand.replace(/[^\d.]/g, ''));
        if (num > 0) {
            document.getElementById('takt-demand').value = num;
            calcTakt();
        }
    }
}

function loadVSMIntoOEE(steps) {
    // Use the first step with data
    const step = steps.find(s => s.cycle_time || s.uptime) || steps[0];
    if (!step) return;
    if (step.uptime) {
        document.getElementById('oee-uptime').value = step.uptime;
    }
    if (step.cycle_time) {
        document.getElementById('oee-ideal').value = step.cycle_time;
    }
    if (typeof calcOEE === 'function') calcOEE();
}

function loadVSMIntoKanbanSizing(vsm) {
    if (vsm.customer_demand) {
        const num = parseFloat(vsm.customer_demand.replace(/[^\d.]/g, ''));
        if (num > 0) document.getElementById('kanban-demand').value = num;
    }
    // Lead time from inventory days of supply
    const invs = vsm.inventory || [];
    if (invs.length > 0) {
        const avgDays = invs.reduce((s, inv) => s + (inv.days_of_supply || 0), 0) / invs.length;
        if (avgDays > 0) document.getElementById('kanban-lead').value = avgDays.toFixed(1);
    }
    if (typeof calcKanban === 'function') calcKanban();
}

function loadVSMIntoCapacityLoad(stations, vsm) {
    if (stations.length === 0) return;
    const demand = parseFloat((vsm.customer_demand || '').replace(/[^\d.]/g, '')) || 100;
    capacityOrders.length = 0;
    stations.forEach((s, i) => {
        capacityOrders.push({
            id: i + 1,
            name: s.name,
            hours: parseFloat(((s.cycle_time * demand) / 3600).toFixed(2)),
            startDay: 1
        });
    });
    if (typeof renderCapacityOrders === 'function') renderCapacityOrders();
    if (typeof calcCapacityLoad === 'function') calcCapacityLoad();
}

function loadVSMIntoRTO(stations, vsm) {
    if (stations.length === 0) return;
    const totalCT = stations.reduce((sum, s) => sum + (s.cycle_time || 0), 0);
    document.getElementById('rto-cycle').value = totalCT.toFixed(1);
    if (vsm.takt_time) document.getElementById('rto-takt').value = vsm.takt_time;
    if (typeof calcRTO === 'function') calcRTO();
}

// --- Export Takt to VSM ---

async function exportTaktToVSM() {
    const taktSec = SvendOps.get('takt');
    if (!taktSec) { alert('Calculate takt time first'); return; }

    try {
        const resp = await fetch('/api/vsm/', { credentials: 'same-origin' });
        const data = await resp.json();
        const maps = data.maps || [];
        if (maps.length === 0) { alert('No VSMs found. Create a Value Stream Map first.'); return; }

        // Single VSM — export directly
        if (maps.length === 1) {
            await doExportTaktToVSM(maps[0], taktSec);
            return;
        }

        // Multiple VSMs — show selection modal
        const opts = maps.map(m => `<option value="${m.id}">${m.name}</option>`).join('');
        const modal = document.getElementById('vsm-import-modal');
        modal.querySelector('h3').textContent = 'Export Takt Time to VSM';
        const selectEl = document.getElementById('vsm-import-select');
        selectEl.innerHTML = opts;
        document.getElementById('vsm-import-preview').style.display = 'none';
        const importBtn = document.getElementById('vsm-import-btn');
        importBtn.textContent = 'Export to Selected VSM';
        importBtn.onclick = async () => {
            const selectedId = selectEl.value;
            const target = maps.find(m => String(m.id) === selectedId);
            if (!target) return;
            closeVSMImport();
            await doExportTaktToVSM(target, taktSec);
            importBtn.onclick = () => doVSMImport();
            importBtn.textContent = 'Import into Current Calculator';
            modal.querySelector('h3').textContent = 'Import from Value Stream Map';
        };
        modal.style.display = 'flex';
    } catch (e) {
        alert('Failed to export: ' + safeStr(e, 'Unknown error'));
    }
}

async function doExportTaktToVSM(target, taktSec) {
    try {
        const resp = await fetch(`/api/vsm/${target.id}/update/`, {
            method: 'PUT',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ takt_time: taktSec })
        });
        if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
        alert(`Takt time (${taktSec.toFixed(1)}s) exported to "${target.name}"`);
    } catch (e) {
        alert('Failed to export: ' + safeStr(e, 'Unknown error'));
    }
}

// ============================================================================
// Generic VSM Export (bidirectional sync)
// ============================================================================

async function exportToVSM(payload, successMsg) {
    try {
        const resp = await fetch('/api/vsm/', { credentials: 'same-origin' });
        const data = await resp.json();
        const maps = (data.maps || []).filter(m => m.status === 'current');
        if (maps.length === 0) { showToast('No VSMs found. Create a Value Stream Map first.', 'warning'); return; }

        if (maps.length === 1) {
            await doGenericExportToVSM(maps[0], payload);
            showToast(successMsg || 'Exported to VSM');
            return;
        }

        // Multiple VSMs — reuse import modal for selection
        const modal = document.getElementById('vsm-import-modal');
        modal.querySelector('h3').textContent = 'Export Results to VSM';
        const selectEl = document.getElementById('vsm-import-select');
        selectEl.innerHTML = maps.map(m => `<option value="${m.id}">${m.name} (${m.process_steps?.length || 0} steps)</option>`).join('');
        document.getElementById('vsm-import-preview').style.display = 'none';
        const btn = document.getElementById('vsm-import-btn');
        btn.textContent = 'Export to Selected VSM';
        btn.onclick = async () => {
            const id = selectEl.value;
            const target = maps.find(m => String(m.id) === id);
            if (!target) return;
            closeVSMImport();
            await doGenericExportToVSM(target, payload);
            showToast(successMsg || 'Exported to VSM');
            // Restore modal to import mode
            btn.onclick = () => doVSMImport();
            btn.textContent = 'Import into Current Calculator';
            modal.querySelector('h3').textContent = 'Import from Value Stream Map';
        };
        modal.style.display = 'flex';
    } catch (e) {
        showToast('Export failed: ' + safeStr(e, 'Unknown error'), 'error');
    }
}

async function doGenericExportToVSM(target, payload) {
    // Fetch current VSM, merge annotations into matching steps, PUT back
    const resp = await fetch(`/api/vsm/${target.id}/`, { credentials: 'same-origin' });
    if (!resp.ok) throw new Error('Failed to fetch VSM');
    const data = await resp.json();
    const vsm = data.vsm;

    // Merge step-level annotations
    if (payload.step_annotations) {
        for (const [stepName, annotation] of Object.entries(payload.step_annotations)) {
            const step = vsm.process_steps.find(s => s.name === stepName);
            if (!step) continue;
            if (!step.annotations) step.annotations = [];
            // Replace existing annotation from same source, or append
            const idx = step.annotations.findIndex(a => a.source === annotation.source);
            if (idx >= 0) step.annotations[idx] = annotation;
            else step.annotations.push(annotation);
        }
    }

    // Merge VSM-level fields (e.g. takt_time)
    const putPayload = { ...(payload.vsm_fields || {}) };
    putPayload.process_steps = vsm.process_steps;

    // Handle auto kaizen burst creation
    if (payload.auto_kaizen) {
        const ak = payload.auto_kaizen;
        const bursts = vsm.kaizen_bursts || [];
        const exists = bursts.some(b => b.text === ak.text);
        if (!exists) {
            // Position near the named step
            const nearStep = vsm.process_steps.find(s => s.name === ak.near_step);
            const x = nearStep ? (nearStep.x || 200) + 20 : 200;
            const y = nearStep ? (nearStep.y || 200) - 60 : 200;
            bursts.push({
                id: Math.random().toString(36).substring(2, 10),
                x, y,
                text: ak.text,
                priority: ak.priority || 'medium'
            });
            putPayload.kaizen_bursts = bursts;
        }
    }

    const putResp = await fetch(`/api/vsm/${target.id}/update/`, {
        method: 'PUT', credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(putPayload)
    });
    if (!putResp.ok) throw new Error(`Server returned ${putResp.status}`);
}

// --- Calculator-specific VSM exports ---

async function exportBottleneckToVSM() {
    if (bottleneckData.length === 0) { showToast('Run bottleneck analysis first', 'warning'); return; }
    const maxTime = Math.max(...bottleneckData.map(s => s.time));
    const constraint = bottleneckData.find(s => s.time === maxTime);
    const annotations = {};
    bottleneckData.forEach(s => {
        annotations[s.name] = {
            source: 'bottleneck', type: 'bottleneck_analysis',
            value: { cycle_time: s.time, is_constraint: s.time === maxTime, throughput: parseFloat((3600 / s.time).toFixed(1)) },
            status: s.time === maxTime ? 'red' : 'green',
            timestamp: new Date().toISOString()
        };
    });
    await exportToVSM({
        step_annotations: annotations,
        auto_kaizen: { text: `Bottleneck: ${constraint.name} (${constraint.time}s — ${(3600/constraint.time).toFixed(0)} u/hr)`, near_step: constraint.name, priority: 'high' }
    }, `Bottleneck exported. Constraint: ${constraint.name}`);
}

async function exportOEEToVSM() {
    const oeeVal = SvendOps.get('oee');
    if (!oeeVal) { showToast('Calculate OEE first', 'warning'); return; }
    // Determine which step — if imported from VSM, use that context
    const availability = document.getElementById('oee-availability')?.textContent || '';
    const performance = document.getElementById('oee-performance')?.textContent || '';
    const quality = document.getElementById('oee-quality')?.textContent || '';

    // Show step selector
    try {
        const resp = await fetch('/api/vsm/', { credentials: 'same-origin' });
        const data = await resp.json();
        const maps = (data.maps || []).filter(m => m.status === 'current');
        if (maps.length === 0) { showToast('No VSMs found', 'warning'); return; }
        const vsm = maps[0]; // use first; multi-select can be added later
        const steps = vsm.process_steps || [];
        if (steps.length === 0) { showToast('VSM has no process steps', 'warning'); return; }
        const stepName = await selectVSMStep(steps, 'Which step does this OEE apply to?');
        if (!stepName) return;
        const annotations = {};
        annotations[stepName] = {
            source: 'oee', type: 'oee_result',
            value: { oee: oeeVal, availability, performance, quality },
            status: oeeVal >= 85 ? 'green' : oeeVal >= 65 ? 'yellow' : 'red',
            timestamp: new Date().toISOString()
        };
        await exportToVSM({ step_annotations: annotations }, `OEE (${oeeVal.toFixed(1)}%) exported to "${stepName}"`);
    } catch (e) { showToast('Export failed: ' + safeStr(e, 'Unknown error'), 'error'); }
}

async function exportLineSimToVSM() {
    if (lineStations.length === 0 || !lineSimState || lineSimState.time === 0) {
        showToast('Run line simulation first', 'warning'); return;
    }
    const annotations = {};
    const stats = lineSimState.stationStats || [];
    lineStations.forEach((s, i) => {
        const st = stats[i];
        if (!st) return;
        const total = st.working + st.blocked + st.starved;
        const utilization = total > 0 ? parseFloat(((st.working / total) * 100).toFixed(1)) : 0;
        annotations[s.name] = {
            source: 'line_sim', type: 'utilization',
            value: { utilization, blocked_pct: total > 0 ? parseFloat(((st.blocked / total) * 100).toFixed(1)) : 0, starved_pct: total > 0 ? parseFloat(((st.starved / total) * 100).toFixed(1)) : 0 },
            status: utilization > 95 ? 'red' : utilization > 80 ? 'yellow' : 'green',
            timestamp: new Date().toISOString()
        };
    });
    await exportToVSM({ step_annotations: annotations }, `Line sim utilization (${lineStations.length} stations) exported`);
}

async function exportTOCToVSM() {
    if (typeof tocStationsData === 'undefined' || tocStationsData.length === 0) { showToast('Run TOC analysis first', 'warning'); return; }
    const minCap = Math.min(...tocStationsData.map(s => s.capacity));
    const constraint = tocStationsData.find(s => s.capacity === minCap);
    const annotations = {};
    tocStationsData.forEach(s => {
        annotations[s.name] = {
            source: 'toc', type: 'toc_analysis',
            value: { capacity: s.capacity, is_constraint: s.capacity === minCap },
            status: s.capacity === minCap ? 'red' : 'green',
            timestamp: new Date().toISOString()
        };
    });
    await exportToVSM({ step_annotations: annotations }, `TOC constraint: ${constraint.name} (${constraint.capacity} u/hr)`);
}

async function exportCellSimToVSM() {
    if (lineStations.length === 0) { showToast('Configure cell design first', 'warning'); return; }
    const annotations = {};
    lineStations.forEach(s => {
        annotations[s.name] = {
            source: 'cell_sim', type: 'cell_design',
            value: { cycle_time: s.cycleTime },
            status: 'green',
            timestamp: new Date().toISOString()
        };
    });
    await exportToVSM({ step_annotations: annotations }, `Cell design (${lineStations.length} stations) exported`);
}

async function exportKanbanSimToVSM() {
    if (typeof kanbanStations === 'undefined' || kanbanStations.length === 0) { showToast('Run Kanban sim first', 'warning'); return; }
    const annotations = {};
    kanbanStations.forEach(s => {
        annotations[s.name] = {
            source: 'kanban_sim', type: 'kanban_sim',
            value: { cycle_time: s.cycleTime },
            status: 'green',
            timestamp: new Date().toISOString()
        };
    });
    await exportToVSM({ step_annotations: annotations }, `Kanban sim (${kanbanStations.length} stations) exported`);
}

async function exportSafetySimToVSM() {
    if (typeof safetySimStations === 'undefined' || safetySimStations.length === 0) { showToast('Run safety stock sim first', 'warning'); return; }
    const annotations = {};
    safetySimStations.forEach(s => {
        annotations[s.name] = {
            source: 'safety_sim', type: 'safety_stock',
            value: { cycle_time: s.cycleTime },
            status: 'green',
            timestamp: new Date().toISOString()
        };
    });
    await exportToVSM({ step_annotations: annotations }, 'Safety stock sim exported');
}

async function exportCapacityLoadToVSM() {
    if (typeof capacityOrders === 'undefined' || capacityOrders.length === 0) { showToast('Run capacity load first', 'warning'); return; }
    const annotations = {};
    capacityOrders.forEach(o => {
        annotations[o.name] = {
            source: 'capacity_load', type: 'capacity_load',
            value: { load_hours: o.hours },
            status: o.hours > 8 ? 'red' : o.hours > 6 ? 'yellow' : 'green',
            timestamp: new Date().toISOString()
        };
    });
    await exportToVSM({ step_annotations: annotations }, `Capacity load (${capacityOrders.length} stations) exported`);
}

async function exportRTOToVSM() {
    const staffEl = document.getElementById('rto-staff');
    if (!staffEl) { showToast('Run RTO calculation first', 'warning'); return; }
    const staff = parseInt(staffEl.textContent);
    if (isNaN(staff)) { showToast('Calculate RTO first', 'warning'); return; }
    await exportToVSM({
        vsm_fields: {},
        step_annotations: {}
    }, `RTO staffing recommendation: ${staff} people`);
}

async function exportKanbanSizingToVSM() {
    const result = document.getElementById('kanban-result');
    if (!result) { showToast('Calculate kanban sizing first', 'warning'); return; }
    const qty = parseInt(result.textContent);
    if (isNaN(qty)) { showToast('Calculate kanban first', 'warning'); return; }
    showToast(`Kanban sizing: ${qty} cards. Data is informational — no step-level export.`);
}

async function exportEPEIToVSM() {
    const result = document.getElementById('epei-result');
    if (!result) { showToast('Calculate EPEI first', 'warning'); return; }
    showToast(`EPEI result noted. Data is informational — no step-level export.`);
}

// Step selector helper for single-step exports
function selectVSMStep(steps, prompt_msg) {
    return new Promise((resolve) => {
        const modal = document.getElementById('vsm-import-modal');
        modal.querySelector('h3').textContent = prompt_msg || 'Select Process Step';
        const selectEl = document.getElementById('vsm-import-select');
        selectEl.innerHTML = steps.map(s => `<option value="${s.name}">${s.name} (C/T: ${s.cycle_time ? s.cycle_time + 's' : '-'})</option>`).join('');
        document.getElementById('vsm-import-preview').style.display = 'none';
        const btn = document.getElementById('vsm-import-btn');
        btn.textContent = 'Select Step';
        btn.onclick = () => {
            const selected = selectEl.value;
            closeVSMImport();
            btn.onclick = () => doVSMImport();
            btn.textContent = 'Import into Current Calculator';
            modal.querySelector('h3').textContent = 'Import from Value Stream Map';
            resolve(selected || null);
        };
        modal.style.display = 'flex';
    });
}

// --- Pin to VSM (DSW / SPC results) ---

async function pinDSWResultToVSM(summary, resultType, status) {
    const annotation = {
        source: 'dsw_pin',
        type: resultType || 'dsw_result',
        value: { summary: summary },
        status: status || 'green',
        timestamp: new Date().toISOString()
    };
    try {
        const resp = await fetch('/api/vsm/', { credentials: 'same-origin' });
        const data = await resp.json();
        const maps = (data.maps || []).filter(m => m.status === 'current');
        if (maps.length === 0) { showToast('No VSMs found', 'warning'); return; }
        const target = maps.length === 1 ? maps[0] : await new Promise(resolve => {
            exportToVSM({ step_annotations: {} }, '');  // reuse modal; will override below
            return;
        });
        const vsmResp = await fetch(`/api/vsm/${(maps[0]).id}/`, { credentials: 'same-origin' });
        const vsmData = await vsmResp.json();
        const steps = (vsmData.vsm.process_steps || []).filter(s => s.name);
        if (steps.length === 0) { showToast('VSM has no named steps', 'warning'); return; }
        const stepName = await selectVSMStep(steps, 'Pin DSW result to which step?');
        if (!stepName) return;
        await doGenericExportToVSM(maps.length === 1 ? maps[0] : maps[0], {
            step_annotations: { [stepName]: annotation }
        });
        showToast('Pinned to VSM step: ' + stepName);
    } catch (e) {
        showToast('Pin failed: ' + safeStr(e, 'Unknown error'), 'error');
    }
}

async function pinSPCResultToVSM(summary, controlStatus) {
    const statusMap = { in_control: 'green', warning: 'yellow', out_of_control: 'red' };
    const annotation = {
        source: 'spc_pin',
        type: 'spc_result',
        value: { summary: summary, control_status: controlStatus || 'unknown' },
        status: statusMap[controlStatus] || 'yellow',
        timestamp: new Date().toISOString()
    };
    try {
        const resp = await fetch('/api/vsm/', { credentials: 'same-origin' });
        const data = await resp.json();
        const maps = (data.maps || []).filter(m => m.status === 'current');
        if (maps.length === 0) { showToast('No VSMs found', 'warning'); return; }
        const vsmResp = await fetch(`/api/vsm/${maps[0].id}/`, { credentials: 'same-origin' });
        const vsmData = await vsmResp.json();
        const steps = (vsmData.vsm.process_steps || []).filter(s => s.name);
        if (steps.length === 0) { showToast('VSM has no named steps', 'warning'); return; }
        const stepName = await selectVSMStep(steps, 'Pin SPC result to which step?');
        if (!stepName) return;
        await doGenericExportToVSM(maps[0], { step_annotations: { [stepName]: annotation } });
        showToast('Pinned SPC result to: ' + stepName);
    } catch (e) {
        showToast('Pin failed: ' + safeStr(e, 'Unknown error'), 'error');
    }
}
