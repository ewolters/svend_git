// =============================================================================
// Utilities
// =============================================================================

function getCsrf() {
    return svendGetCsrf();
}

function showToast(msg) {
    svendToast(msg, 2500);
}

// Speed slider label
document.getElementById('cfg-speed').addEventListener('input', function() {
    document.getElementById('cfg-speed-label').textContent = this.value + 'x';
});

// =============================================================================
// Workforce Management
// =============================================================================

const OPERATOR_NAMES = [
    'Martinez', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
    'Davis', 'Rodriguez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore',
    'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson', 'White', 'Harris',
    'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker', 'Young',
];

let opNameIdx = 0;

function addOperator(role = 'operator') {
    const name = OPERATOR_NAMES[opNameIdx++ % OPERATOR_NAMES.length];
    // Skills: map of station_id -> skill level (0.0 to 1.0)
    const skills = {};
    const startSkill = role === 'operator' ? 0.3 : 0;
    for (const stn of layout.stations) {
        skills[stn.id] = startSkill;
    }
    layout.operators.push({
        id: genId('op'),
        name,
        role, // operator, maintenance, agv_driver, inspector
        shift: 1, // Shift 1, 2, or 3
        primaryStation: null, // explicit machine assignment (null = floating)
        skills, // { station_id: 0.0-1.0 }
        status: 'available', // available, busy, absent, training, quit
        assignedTo: null, // station_id currently working at
    });
    renderOperatorList();
}

function removeOperator() {
    if (layout.operators.length === 0) return;
    layout.operators.pop();
    renderOperatorList();
}

function renderOperatorList() {
    const list = document.getElementById('operator-list');
    list.innerHTML = '';
    const roleLabels = { operator: 'Op', maintenance: 'Mt', agv_driver: 'AGV', inspector: 'Insp' };
    for (const op of layout.operators) {
        const row = document.createElement('div');
        row.className = 'operator-row' + (op.status === 'absent' || op.status === 'quit' ? ' absent' : '');
        const bestSkill = Object.values(op.skills || {}).reduce((a, b) => Math.max(a, b), 0);
        const skillBar = Math.round(bestSkill * 100);
        const desOp = des ? des.operators.find(o => o.id === op.id) : null;
        const fatiguePct = desOp ? Math.round(desOp.fatigue * 100) : 0;
        const moralePct = desOp ? Math.round(desOp.morale * 100) : 0;
        const fatigueColor = fatiguePct > 60 ? '#e74c3c' : fatiguePct > 30 ? '#f59e0b' : 'var(--text-dim)';
        const moraleColor = moralePct < 50 ? '#e74c3c' : moralePct < 70 ? '#f59e0b' : 'var(--text-dim)';
        const moraleIndicator = desOp ? `<span style="font-size:0.5rem;color:${moraleColor};" title="Morale ${moralePct}% / Fatigue ${fatiguePct}%"><svg width="7" height="7" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" style="vertical-align:middle"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>${moralePct}</span>` : '';
        const roleTag = roleLabels[op.role || 'operator'] || 'Op';
        const shiftTag = op.shift ? `S${op.shift}` : '';
        row.innerHTML = `
            <span class="operator-name" title="${op.name}">${op.name}</span>
            <span style="font-size:0.45rem;color:var(--accent-primary);padding:0 2px;border:1px solid var(--border);border-radius:2px;">${roleTag}</span>
            ${shiftTag ? `<span style="font-size:0.45rem;color:var(--text-dim);">${shiftTag}</span>` : ''}
            ${moraleIndicator}
            <span style="font-size:0.6rem;color:${fatigueColor};">${desOp ? fatiguePct + 'F' : skillBar + '%'}</span>
            <span class="operator-status ${op.status}">${op.status}</span>
        `;
        row.onclick = () => showOperatorDetail(op);
        list.appendChild(row);
    }
    const available = layout.operators.filter(o => o.status === 'available' || o.status === 'busy').length;
    const total = layout.operators.length;
    document.getElementById('workforce-stat').textContent = `${available}/${total} available`;
    updateRTODisplay();
}

function updateRTODisplay() {
    const el = document.getElementById('rto-display');
    if (!el) return;
    const ops = layout.operators.filter(o => o.status !== 'quit');
    const byRole = { operator: 0, maintenance: 0, agv_driver: 0, inspector: 0 };
    for (const op of ops) byRole[op.role || 'operator']++;
    const machines = layout.stations.filter(s => s.type !== 'source' && s.type !== 'sink');
    const reqOps = machines.reduce((sum, m) => sum + (m.operators || 1), 0);
    const reqMaint = +(document.getElementById('cfg-maint-crew')?.value || 0);
    const reqAGV = +(document.getElementById('cfg-agv-fleet')?.value || 0);
    const reqInsp = +(document.getElementById('cfg-inspector-pool')?.value || 0);
    const totalReq = reqOps + reqMaint + reqAGV + reqInsp;
    const totalHave = ops.length;
    const short = Math.max(0, totalReq - totalHave);
    const lines = [`<strong>RTO: ${totalHave} / ${totalReq}${short > 0 ? ` (${short} short)` : ''}</strong>`];
    const check = (have, need, label) => {
        const diff = need - have;
        return `${label}: ${have}/${need} ${diff > 0 ? `(${diff} short)` : ''}`;
    };
    lines.push(check(byRole.operator, reqOps, 'Operators'));
    if (reqMaint > 0) lines.push(check(byRole.maintenance, reqMaint, 'Maintenance'));
    if (reqAGV > 0) lines.push(check(byRole.agv_driver, reqAGV, 'AGV'));
    if (reqInsp > 0) lines.push(check(byRole.inspector, reqInsp, 'Inspectors'));
    el.innerHTML = lines.join('<br>');
}

function showOperatorDetail(op) {
    const panel = document.getElementById('props-panel');
    const content = document.getElementById('props-content');
    panel.classList.add('active');
    selectedElement = op;

    let skillRows = '';
    for (const stn of layout.stations) {
        const skill = op.skills[stn.id] ?? 0;
        const pct = (skill * 100).toFixed(0);
        const color = skill >= 0.8 ? '#4a9f6e' : skill >= 0.5 ? '#f59e0b' : '#e74c3c';
        const hasTrainer = layout.operators.some(o => o.id !== op.id && o.status !== 'quit' && (o.skills[stn.id] || 0) > 0.3);
        const trainBtn = skill < 0.9 && op.status === 'available'
            ? `<button onclick="startCrossTraining('${op.id}', '${stn.id}')" style="font-size:0.5rem;padding:0 4px;background:var(--bg-tertiary);border:1px solid var(--border);color:var(--text-primary);cursor:pointer;border-radius:2px;margin-left:2px;" ${hasTrainer ? '' : 'disabled title="No experienced trainer available"'}>${hasTrainer ? 'Train' : 'No trainer'}</button>`
            : '';
        skillRows += `
            <div class="prop-row">
                <span class="prop-label" style="font-size:0.65rem;">${stn.name}</span>
                <div style="flex:1;background:var(--bg-primary);height:8px;border-radius:4px;overflow:hidden;">
                    <div style="width:${pct}%;height:100%;background:${color};border-radius:4px;"></div>
                </div>
                <span style="font-size:0.6rem;color:var(--text-dim);width:30px;text-align:right;">${pct}%</span>
                ${trainBtn}
            </div>`;
    }

    const desOp = des ? des.operators.find(o => o.id === op.id) : null;
    const statsHtml = desOp ? `
        <div style="margin-top:6px;border-top:1px solid var(--border);padding-top:4px;">
            <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;color:var(--text-dim);margin-bottom:4px;">Stats</div>
            <div style="font-size:0.6rem;color:var(--text-dim);line-height:1.6;">
                Parts: ${desOp.partsProcessed || 0} · Fatigue: ${(desOp.fatigue * 100).toFixed(0)}% · Morale: ${(desOp.morale * 100).toFixed(0)}%
            </div>
        </div>` : '';

    content.innerHTML = `
        <div class="prop-row"><span class="prop-label">Name</span><input class="prop-input" value="${op.name}" onchange="selectedElement.name = this.value; renderOperatorList();"></div>
        <div class="prop-row"><span class="prop-label">Role</span>
            <select class="prop-input" onchange="selectedElement.role = this.value; renderOperatorList();">
                <option value="operator" ${(op.role || 'operator') === 'operator' ? 'selected' : ''}>Operator</option>
                <option value="maintenance" ${op.role === 'maintenance' ? 'selected' : ''}>Maintenance</option>
                <option value="agv_driver" ${op.role === 'agv_driver' ? 'selected' : ''}>AGV Driver</option>
                <option value="inspector" ${op.role === 'inspector' ? 'selected' : ''}>Inspector</option>
            </select>
        </div>
        <div class="prop-row"><span class="prop-label">Shift</span>
            <select class="prop-input" onchange="selectedElement.shift = +this.value; renderOperatorList();">
                <option value="1" ${(op.shift || 1) === 1 ? 'selected' : ''}>Shift 1</option>
                <option value="2" ${op.shift === 2 ? 'selected' : ''}>Shift 2</option>
                <option value="3" ${op.shift === 3 ? 'selected' : ''}>Shift 3</option>
            </select>
        </div>
        <div class="prop-row"><span class="prop-label">Machine</span>
            <select class="prop-input" onchange="selectedElement.primaryStation = this.value || null; renderOperatorList();">
                <option value="" ${!op.primaryStation ? 'selected' : ''}>Floating (best match)</option>
                ${layout.stations.filter(s => s.type !== 'source' && s.type !== 'sink').map(s =>
                    `<option value="${s.id}" ${op.primaryStation === s.id ? 'selected' : ''}>${s.name || s.id}</option>`
                ).join('')}
            </select>
        </div>
        <div style="font-size:0.65rem;color:var(--text-dim);margin:4px 0 8px 0;">Status: ${op.status}${op.assignedTo ? ' at ' + (layout.stations.find(s => s.id === op.assignedTo)?.name || op.assignedTo) : ''}</div>
        <div style="font-size:0.65rem;font-weight:600;text-transform:uppercase;color:var(--text-dim);margin-bottom:4px;">Skills</div>
        ${skillRows || '<div style="font-size:0.7rem;color:var(--text-dim);">No machines placed</div>'}
        ${statsHtml}
    `;
}

function startCrossTraining(opId, stnId) {
    const op = layout.operators.find(o => o.id === opId);
    if (!op) return;
    const stn = layout.stations.find(s => s.id === stnId);
    const trainer = layout.operators.find(o => o.id !== opId && o.status !== 'quit' && (o.skills[stnId] || 0) > 0.3);
    if (!trainer) { showToast('No experienced trainer available (>30% skill required)'); return; }
    op._trainingTarget = stnId;
    op._trainingHours = 4;
    op.status = 'training';
    trainer._trainingTarget = stnId;
    trainer._trainerFor = opId;
    trainer.status = 'training';
    renderOperatorList();
    showToast(`${op.name} training on ${stn?.name || stnId} with ${trainer.name} (4hr)`);
    showOperatorDetail(op);
}

function showTrainingDialog() {
    // CR-5: Use operator detail panel instead of prompt() dialogs
    if (layout.operators.length === 0) { showToast('Add operators first'); return; }
    const availOps = layout.operators.filter(o => o.status === 'available');
    if (availOps.length === 0) { showToast('No available operators — select one from list'); return; }
    showOperatorDetail(availOps[0]);
    showToast('Select a station "Train" button to cross-train');
}

// Init operator list
renderOperatorList();

// =============================================================================
// Shared Tooling / Fixture Management
// =============================================================================

if (!layout.shared_tools) layout.shared_tools = [];

function addSharedTool() {
    const types = ['Fixture A', 'Jig B', 'Die Set C', 'Gauge D', 'Template E'];
    const existing = layout.shared_tools.length;
    layout.shared_tools.push({
        id: 'tool-' + Date.now(),
        name: types[existing % types.length],
        copies: 1,
        required_by: [],
    });
    renderToolList();
}

function removeSharedTool(idx) {
    layout.shared_tools.splice(idx, 1);
    renderToolList();
}

function renderToolList() {
    const container = document.getElementById('tool-list');
    if (!container) return;
    if (layout.shared_tools.length === 0) {
        container.innerHTML = '<div style="color:var(--text-dim);">No shared tools.</div>';
        return;
    }
    const machines = layout.stations || [];
    container.innerHTML = layout.shared_tools.map((t, i) => `
        <div style="margin-bottom:6px; padding:4px; background:var(--bg-tertiary); border-radius:4px;">
            <div style="display:flex; align-items:center; gap:4px; margin-bottom:3px;">
                <input type="text" value="${t.name}" onchange="layout.shared_tools[${i}].name = this.value" style="flex:1; background:var(--bg-primary); border:1px solid var(--border); color:var(--text-primary); padding:1px 4px; font-size:0.6rem; border-radius:3px;">
                <span style="font-size:0.55rem; color:var(--text-dim);">×</span>
                <input type="number" value="${t.copies}" min="1" max="10" style="width:30px; background:var(--bg-primary); border:1px solid var(--border); color:var(--text-primary); padding:1px 2px; font-size:0.6rem; border-radius:3px; text-align:center;" onchange="layout.shared_tools[${i}].copies = +this.value">
                <button onclick="removeSharedTool(${i})" style="background:none; border:none; color:#e74c3c; cursor:pointer; font-size:0.7rem; padding:0 2px;">&#10005;</button>
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:2px;">
                ${machines.map(m => {
                    const checked = (t.required_by || []).includes(m.id);
                    return `<label style="font-size:0.5rem; display:flex; align-items:center; gap:2px; cursor:pointer;">
                        <input type="checkbox" ${checked ? 'checked' : ''} onchange="toggleToolMachine(${i}, '${m.id}', this.checked)" style="width:10px; height:10px;">
                        ${m.name}
                    </label>`;
                }).join('')}
            </div>
        </div>
    `).join('');
}

function toggleToolMachine(toolIdx, stnId, checked) {
    const tool = layout.shared_tools[toolIdx];
    if (!tool.required_by) tool.required_by = [];
    if (checked && !tool.required_by.includes(stnId)) {
        tool.required_by.push(stnId);
    } else if (!checked) {
        tool.required_by = tool.required_by.filter(id => id !== stnId);
    }
}

renderToolList();

// Utility System Management
// =============================================================================

if (!layout.utility_systems) layout.utility_systems = [];

function addUtilitySystem() {
    const types = ['Compressed Air', 'Chilled Water', 'Electrical Bus', 'Steam', 'Hydraulic', 'Vacuum'];
    const existing = layout.utility_systems.length;
    const name = types[existing % types.length];
    layout.utility_systems.push({
        id: genId('util'),
        name,
        mtbf: 7200,    // default 2hr between failures
        mttr: 300,      // default 5min repair
        affected_machines: [],
    });
    renderUtilityList();
}

function removeUtilitySystem(idx) {
    layout.utility_systems.splice(idx, 1);
    renderUtilityList();
}

function updateUtility(idx, field, value) {
    layout.utility_systems[idx][field] = value;
    renderUtilityList();
}

function toggleUtilityMachine(utilIdx, stnId) {
    const util = layout.utility_systems[utilIdx];
    const idx = util.affected_machines.indexOf(stnId);
    if (idx === -1) {
        util.affected_machines.push(stnId);
    } else {
        util.affected_machines.splice(idx, 1);
    }
    renderUtilityList();
}

function renderUtilityList() {
    const list = document.getElementById('utility-list');
    if (!list) return;
    if (layout.utility_systems.length === 0) {
        list.innerHTML = '<div style="color:var(--text-dim); font-size:0.55rem;">No utilities configured</div>';
        return;
    }
    list.innerHTML = layout.utility_systems.map((u, i) => {
        const machineChecks = layout.stations.map(s => {
            const checked = u.affected_machines.includes(s.id) ? 'checked' : '';
            return `<label style="display:inline-flex;align-items:center;gap:2px;margin-right:4px;font-size:0.55rem;cursor:pointer;">
                <input type="checkbox" ${checked} onchange="toggleUtilityMachine(${i}, '${s.id}')" style="width:10px;height:10px;">
                ${s.name || s.id}
            </label>`;
        }).join('');
        // Show live state if DES running
        const desUtil = des ? des.utilitySystems.find(du => du.id === u.id) : null;
        const stateLabel = desUtil ? (desUtil.state === 'down'
            ? '<span style="color:#e74c3c;font-weight:600;">DOWN</span>'
            : '<span style="color:var(--accent-primary);">UP</span>') : '';
        return `
            <div style="border:1px solid var(--border);border-radius:4px;padding:4px;margin-bottom:4px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <input class="prop-input" value="${u.name}" onchange="updateUtility(${i}, 'name', this.value)" style="width:80px;font-size:0.6rem;">
                    ${stateLabel}
                    <button onclick="removeUtilitySystem(${i})" style="font-size:0.6rem;cursor:pointer;background:none;border:none;color:#e74c3c;">×</button>
                </div>
                <div class="prop-row" style="margin-top:2px;">
                    <span class="prop-label" style="width:30px;">MTBF</span>
                    <input type="number" class="prop-input" value="${u.mtbf || ''}" onchange="updateUtility(${i}, 'mtbf', +this.value)" style="width:45px;" placeholder="0">
                    <span class="prop-unit">s</span>
                    <span class="prop-label" style="width:30px;margin-left:4px;">MTTR</span>
                    <input type="number" class="prop-input" value="${u.mttr || ''}" onchange="updateUtility(${i}, 'mttr', +this.value)" style="width:45px;" placeholder="300">
                    <span class="prop-unit">s</span>
                </div>
                <div style="margin-top:2px;">${machineChecks || '<span style="color:var(--text-dim);">Add machines first</span>'}</div>
            </div>`;
    }).join('');
}

renderUtilityList();

