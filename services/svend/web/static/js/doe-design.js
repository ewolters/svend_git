// DOE Workbench — Design Type Selection, Factor Management, Design Generation
// Load order: doe-state.js → doe-design.js → doe-analysis.js → doe-optimize.js → doe-power.js → doe-chat.js

// Load design types from API
async function loadDesignTypes() {
    try {
        const response = await fetch('/api/experimenter/design/types/', { credentials: 'include' });
        const data = await response.json();
        designTypes = data.design_types || [];
        renderDesignTypes();
    } catch (err) {
        console.error('Failed to load design types:', err);
        // Fallback
        designTypes = [
            { id: 'full_factorial', name: 'Full Factorial', category: 'factorial', description: 'All combinations', runs: '2^k' },
            { id: 'fractional_factorial', name: 'Fractional Factorial', category: 'factorial', description: 'Subset of combinations', runs: '2^(k-p)' },
            { id: 'plackett_burman', name: 'Plackett-Burman', category: 'screening', description: 'Screening design', runs: 'N runs' },
            { id: 'ccd', name: 'Central Composite', category: 'rsm', description: 'Response surface', runs: '2^k + 2k + cp' },
            { id: 'box_behnken', name: 'Box-Behnken', category: 'rsm', description: 'RSM without corners', runs: 'Fewer runs' },
        ];
        renderDesignTypes();
    }
}

async function loadProjects() {
    try {
        const response = await fetch('/api/core/projects/', { credentials: 'include' });
        const data = await response.json();
        projects = data.projects || [];

        const select = document.getElementById('project-selector');
        select.innerHTML = '<option value="">No Study Linked</option>' +
            projects.map(p => `<option value="${p.id}">${p.title}</option>`).join('');
    } catch (err) {
        console.error('Failed to load projects:', err);
    }
}

function onProjectChange(projectId) {
    const url = new URL(window.location);
    if (projectId) {
        url.searchParams.set('project', projectId);
    } else {
        url.searchParams.delete('project');
    }
    window.history.replaceState({}, '', url);
}

// Render design type cards
function renderDesignTypes(filter = 'all') {
    const grid = document.getElementById('design-type-grid');
    const filtered = filter === 'all' ? designTypes : designTypes.filter(d => d.category === filter);

    grid.innerHTML = filtered.map(d => `
        <div class="design-type-card ${d.id === selectedDesignType ? 'selected' : ''}"
             onclick="selectDesignType('${d.id}')">
            <div class="name">${d.name}</div>
            <div class="desc">${d.description}</div>
            <div class="runs">${d.runs}</div>
        </div>
    `).join('');
}

function filterDesigns(category) {
    document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    renderDesignTypes(category);
}

function selectDesignType(typeId) {
    selectedDesignType = typeId;
    renderDesignTypes(document.querySelector('.cat-btn.active')?.textContent?.toLowerCase() || 'all');

    const design = designTypes.find(d => d.id === typeId);
    document.getElementById('center-points').parentElement.style.display =
        design?.supports_center_points ? 'inline-block' : 'none';

    // Resolution picker: show for fractional_factorial and full_factorial
    const resGroup = document.getElementById('resolution-group');
    if (resGroup) resGroup.style.display =
        (typeId === 'fractional_factorial' || typeId === 'full_factorial') ? 'inline-block' : 'none';

    // D-optimal / I-optimal conditional fields
    const optimalOpts = document.getElementById('optimal-design-options');
    if (optimalOpts) optimalOpts.style.display =
        (typeId === 'd_optimal' || typeId === 'i_optimal') ? 'flex' : 'none';

    // Taguchi array selector
    const taguchiOpts = document.getElementById('taguchi-options');
    if (taguchiOpts) taguchiOpts.style.display =
        typeId === 'taguchi' ? 'flex' : 'none';

    // Show design guidance
    const guidance = document.getElementById('design-guidance');
    if (guidance && design) {
        guidance.textContent = design.when_to_use || '';
    }

    updateRunEstimate();
}

function setResolution(value) {
    document.getElementById('resolution').value = value;
    document.querySelectorAll('.res-btn').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.value) === value);
    });
    updateRunEstimate();
}

function updateRunEstimate() {
    const el = document.getElementById('run-estimate');
    if (!el) return;

    // Count factors and their levels
    const factorRows = document.querySelectorAll('.factor-row');
    const levelCounts = [];
    factorRows.forEach(row => {
        const levelsStr = row.querySelector('.factor-levels').value.trim();
        if (levelsStr) {
            const n = levelsStr.split(',').filter(l => l.trim()).length;
            if (n >= 2) levelCounts.push(n);
        }
    });

    const k = levelCounts.length;
    if (k === 0) { el.textContent = ''; return; }

    const reps = parseInt(document.getElementById('replicates').value) || 1;
    const cp = parseInt(document.getElementById('center-points').value) || 0;
    const res = parseInt(document.getElementById('resolution').value) || 4;
    const dt = selectedDesignType;

    let runs = 0;
    const allTwo = levelCounts.every(n => n === 2);

    if (dt === 'full_factorial') {
        runs = levelCounts.reduce((a, b) => a * b, 1);
    } else if (dt === 'fractional_factorial' && allTwo) {
        // 2^(k-p): resolution determines p
        // Res III: p = k-2 (min 4 runs), Res IV: p = k-3, Res V: p = k-4
        const minExp = res === 3 ? 2 : res === 4 ? 3 : 4;
        const exp = Math.max(minExp, Math.min(k, k));
        const p = Math.max(0, k - (res === 3 ? Math.ceil(Math.log2(k + 1)) : res === 4 ? Math.ceil(Math.log2(k + 1)) + 1 : k));
        runs = Math.pow(2, Math.max(2, k - p));
    } else if (dt === 'plackett_burman') {
        // Nearest multiple of 4 above k+1
        runs = Math.ceil((k + 1) / 4) * 4;
    } else if (dt === 'ccd') {
        // 2^k + 2k + center_points (factorial + axial + center)
        const factorial = allTwo ? Math.pow(2, k) : levelCounts.reduce((a, b) => a * b, 1);
        runs = factorial + 2 * k + Math.max(cp, 1);
    } else if (dt === 'box_behnken') {
        // Approximate: standard BB run counts for k=3..7
        const bb = { 3: 12, 4: 24, 5: 40, 6: 48, 7: 56 };
        runs = bb[k] || (k * (k - 1) + 1);
    } else if (dt === 'taguchi') {
        // Nearest standard Taguchi array
        if (allTwo) runs = Math.pow(2, Math.ceil(Math.log2(k + 1)));
        else runs = levelCounts.reduce((a, b) => a * b, 1); // fallback
    } else if (dt === 'd_optimal' || dt === 'i_optimal') {
        const userRuns = parseInt(document.getElementById('optimal-num-runs')?.value);
        runs = (!isNaN(userRuns) && userRuns > 0) ? userRuns : Math.max(k * 2 + 1, 12);
    } else if (dt === 'latin_square' && k >= 2) {
        const maxLevel = Math.max(...levelCounts);
        runs = maxLevel * maxLevel;
    } else {
        // Generic fallback
        runs = levelCounts.reduce((a, b) => a * b, 1);
    }

    // Apply replicates and center points
    const total = runs * reps + cp;
    el.textContent = `≈ ${total} runs`;
    el.style.color = total > 100 ? 'var(--error, #ef4444)' : total > 50 ? 'var(--warning, #f59e0b)' : 'var(--text-dim)';

    validateFactors();
    updateLivePreview(k, levelCounts, total, runs, reps, cp);
}

// Factor management
function addFactor() {
    const container = document.getElementById('factors-container');
    const index = container.children.length + 1;

    const row = document.createElement('div');
    row.className = 'factor-row';
    row.innerHTML = `
        <input type="text" placeholder="Factor ${index}" class="factor-name">
        <div class="factor-levels-wrap">
            <input type="text" placeholder="e.g. 100, 150, 200" class="factor-levels"
                   oninput="onLevelsInput(this)" onblur="onLevelsInput(this)">
            <div class="factor-chips"></div>
            <div class="level-count-badge"></div>
            <div class="level-quick-actions">
                <button class="level-quick-btn" onclick="quickLevels(this, 2)" title="Set 2 levels">2-lvl</button>
                <button class="level-quick-btn" onclick="quickLevels(this, 3)" title="Set 3 levels">3-lvl</button>
                <button class="level-quick-btn" onclick="toggleRangeHelper(this)" title="Enter as range">Lo‑Hi</button>
            </div>
            <div class="range-helper">
                <label>Low</label>
                <input type="number" class="range-low" placeholder="Low" step="any" oninput="rangeToLevels(this)">
                <label>High</label>
                <input type="number" class="range-high" placeholder="High" step="any" oninput="rangeToLevels(this)">
                <label style="display:flex;align-items:center;gap:3px;">
                    <input type="checkbox" class="range-mid" onchange="rangeToLevels(this)"> Mid
                </label>
            </div>
        </div>
        <div class="factor-type-toggle" data-type="numeric">
            <button class="factor-type-btn active" onclick="setFactorType(this, 'numeric')" title="Numeric">Num</button>
            <button class="factor-type-btn" onclick="setFactorType(this, 'categorical')" title="Categorical">Cat</button>
        </div>
        <input type="text" placeholder="Units" class="factor-units">
        <button class="factor-remove" onclick="this.closest('.factor-row').remove(); updateRunEstimate()">&times;</button>
    `;
    container.appendChild(row);
    updateRunEstimate();
}

// === Level chip rendering ===
function onLevelsInput(input) {
    const wrap = input.closest('.factor-levels-wrap');
    const chipsContainer = wrap.querySelector('.factor-chips');
    const badge = wrap.querySelector('.level-count-badge');
    const raw = input.value.trim();

    if (!raw) {
        chipsContainer.innerHTML = '';
        badge.textContent = '';
        updateRunEstimate();
        return;
    }

    const parts = raw.split(',').map(s => s.trim()).filter(s => s.length > 0);

    chipsContainer.innerHTML = parts.map((part, i) => {
        const num = parseFloat(part);
        const isNumeric = !isNaN(num) && isFinite(num);
        const cls = isNumeric ? '' : ' categorical';
        const display = isNumeric ? num : part;
        return `<span class="level-chip${cls}" data-index="${i}">${display}<span class="chip-remove" onclick="removeLevel(this, ${i})">&times;</span></span>`;
    }).join('');

    const n = parts.length;
    const hasCateg = parts.some(p => { const v = parseFloat(p); return isNaN(v) || !isFinite(v); });
    const allCateg = parts.every(p => { const v = parseFloat(p); return isNaN(v) || !isFinite(v); });

    let badgeText = `${n} level${n !== 1 ? 's' : ''}`;
    if (allCateg && n > 0) {
        badgeText += ' · categorical';
    } else if (hasCateg) {
        badgeText += ' · mixed types';
    }

    badge.textContent = badgeText;
    badge.className = 'level-count-badge' + (hasCateg && !allCateg ? ' warning' : '');

    // Auto-detect type toggle unless user manually overrode it
    const toggle = wrap.closest('.factor-row').querySelector('.factor-type-toggle');
    if (toggle && !toggle.dataset.userSet) {
        const autoType = hasCateg ? 'categorical' : 'numeric';
        toggle.dataset.type = autoType;
        toggle.querySelectorAll('.factor-type-btn').forEach(b => {
            const btnType = b.textContent.trim() === 'Num' ? 'numeric' : 'categorical';
            b.classList.toggle('active', btnType === autoType);
        });
    }

    updateRunEstimate();
}

function removeLevel(chipRemoveEl, index) {
    const wrap = chipRemoveEl.closest('.factor-levels-wrap');
    const input = wrap.querySelector('.factor-levels');
    const parts = input.value.split(',').map(s => s.trim()).filter(s => s.length > 0);
    parts.splice(index, 1);
    input.value = parts.join(', ');
    onLevelsInput(input);
}

// === Factor type toggle ===
function setFactorType(btn, type) {
    const toggle = btn.closest('.factor-type-toggle');
    toggle.querySelectorAll('.factor-type-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    toggle.dataset.type = type;
    toggle.dataset.userSet = 'true';

    // Re-render chips with updated coloring
    const row = btn.closest('.factor-row');
    const input = row.querySelector('.factor-levels');
    if (input) onLevelsInput(input);
}

// === Quick level buttons ===
function quickLevels(btn, count) {
    const wrap = btn.closest('.factor-levels-wrap');
    const input = wrap.querySelector('.factor-levels');
    const current = input.value.trim();

    const existing = current.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));

    if (existing.length >= 2) {
        const lo = Math.min(...existing);
        const hi = Math.max(...existing);
        if (count === 2) {
            input.value = `${lo}, ${hi}`;
        } else if (count === 3) {
            const mid = +((lo + hi) / 2).toFixed(4);
            input.value = `${lo}, ${mid}, ${hi}`;
        }
    } else {
        input.value = count === 2 ? '-1, 1' : '-1, 0, 1';
    }
    onLevelsInput(input);
}

function toggleRangeHelper(btn) {
    const wrap = btn.closest('.factor-levels-wrap');
    const helper = wrap.querySelector('.range-helper');
    helper.classList.toggle('visible');
    if (helper.classList.contains('visible')) {
        helper.querySelector('.range-low').focus();
    }
}

function rangeToLevels(el) {
    const helper = el.closest('.range-helper');
    const wrap = helper.closest('.factor-levels-wrap');
    const input = wrap.querySelector('.factor-levels');

    const lo = parseFloat(helper.querySelector('.range-low').value);
    const hi = parseFloat(helper.querySelector('.range-high').value);
    const includeMid = helper.querySelector('.range-mid').checked;

    if (!isNaN(lo) && !isNaN(hi)) {
        if (includeMid) {
            const mid = +((lo + hi) / 2).toFixed(4);
            input.value = `${lo}, ${mid}, ${hi}`;
        } else {
            input.value = `${lo}, ${hi}`;
        }
        onLevelsInput(input);
    }
}

// === Design-aware validation ===
function validateFactors() {
    const el = document.getElementById('factor-validation');
    if (!el) return;

    const dt = designTypes.find(d => d.id === selectedDesignType);
    if (!dt) { el.className = 'factor-validation'; return; }

    const factorRows = document.querySelectorAll('.factor-row');
    const levelCounts = [];

    factorRows.forEach(row => {
        const name = row.querySelector('.factor-name').value.trim();
        const levelsStr = row.querySelector('.factor-levels').value.trim();
        if (levelsStr) {
            const n = levelsStr.split(',').filter(l => l.trim()).length;
            if (n >= 1) levelCounts.push(n);
        }
    });

    const k = levelCounts.length;
    const messages = [];

    if (k > 0 && dt.min_factors && k < dt.min_factors) {
        messages.push(`${dt.name} requires at least ${dt.min_factors} factors (currently ${k}).`);
    }
    if (dt.max_factors && k > dt.max_factors) {
        messages.push(`${dt.name} supports at most ${dt.max_factors} factors (currently ${k}).`);
    }
    if (selectedDesignType === 'box_behnken' && levelCounts.some(n => n !== 3)) {
        messages.push('Box-Behnken requires exactly 3 levels per factor.');
    }
    if (['fractional_factorial', 'plackett_burman', 'definitive_screening', 'dsd'].includes(selectedDesignType) && levelCounts.some(n => n !== 2)) {
        messages.push(`${dt.name} works best with exactly 2 levels per factor.`);
    }
    if (selectedDesignType === 'ccd' && levelCounts.some(n => n !== 2)) {
        messages.push('CCD uses 2 base levels per factor (axial points added automatically).');
    }

    if (messages.length === 0) {
        el.className = 'factor-validation';
    } else {
        el.className = 'factor-validation warning';
        el.innerHTML = messages.join('<br>');
    }
}

// === Factor templates ===
const FACTOR_TEMPLATES = [
    {
        name: '2-Factor Screening',
        factors: [
            { name: 'Factor A', levels: '-1, 1', units: '' },
            { name: 'Factor B', levels: '-1, 1', units: '' },
        ],
        designType: 'full_factorial',
    },
    {
        name: '3-Factor (2-level)',
        factors: [
            { name: 'Factor A', levels: '-1, 1', units: '' },
            { name: 'Factor B', levels: '-1, 1', units: '' },
            { name: 'Factor C', levels: '-1, 1', units: '' },
        ],
        designType: 'full_factorial',
    },
    {
        name: 'RSM (3 factors)',
        factors: [
            { name: 'Temperature', levels: '150, 175, 200', units: '°C' },
            { name: 'Pressure', levels: '1, 2, 3', units: 'atm' },
            { name: 'Time', levels: '30, 45, 60', units: 'min' },
        ],
        designType: 'box_behnken',
    },
    {
        name: 'Screening (6 factors)',
        factors: [
            { name: 'Factor A', levels: '-1, 1', units: '' },
            { name: 'Factor B', levels: '-1, 1', units: '' },
            { name: 'Factor C', levels: '-1, 1', units: '' },
            { name: 'Factor D', levels: '-1, 1', units: '' },
            { name: 'Factor E', levels: '-1, 1', units: '' },
            { name: 'Factor F', levels: '-1, 1', units: '' },
        ],
        designType: 'plackett_burman',
    },
    {
        name: 'CCD Optimization',
        factors: [
            { name: 'Speed', levels: '100, 200', units: 'rpm' },
            { name: 'Feed', levels: '0.1, 0.3', units: 'mm/rev' },
            { name: 'Depth', levels: '0.5, 1.5', units: 'mm' },
        ],
        designType: 'ccd',
    },
];

function toggleTemplatePanel() {
    const panel = document.getElementById('factor-templates');
    if (panel.style.display === 'none') {
        panel.style.display = 'flex';
        panel.innerHTML = FACTOR_TEMPLATES.map((t, i) =>
            `<button class="factor-template-chip" onclick="applyTemplate(${i})">${t.name}</button>`
        ).join('');
    } else {
        panel.style.display = 'none';
    }
}

function applyTemplate(index) {
    const template = FACTOR_TEMPLATES[index];
    const container = document.getElementById('factors-container');
    container.innerHTML = '';

    template.factors.forEach(f => {
        addFactor();
        const rows = container.querySelectorAll('.factor-row');
        const row = rows[rows.length - 1];
        row.querySelector('.factor-name').value = f.name;
        row.querySelector('.factor-levels').value = f.levels;
        row.querySelector('.factor-units').value = f.units;
        onLevelsInput(row.querySelector('.factor-levels'));
    });

    if (template.designType) {
        selectDesignType(template.designType);
    }

    document.getElementById('factor-templates').style.display = 'none';
    updateRunEstimate();
}

// Generate design
async function generateDesign() {
    const factors = [];
    document.querySelectorAll('.factor-row').forEach(row => {
        const name = row.querySelector('.factor-name').value.trim();
        const levelsStr = row.querySelector('.factor-levels').value.trim();
        const units = row.querySelector('.factor-units').value.trim();

        if (name && levelsStr) {
            const levels = levelsStr.split(',').map(l => {
                const trimmed = l.trim();
                const num = parseFloat(trimmed);
                return isNaN(num) ? trimmed : num;
            });

            const typeToggle = row.querySelector('.factor-type-toggle');
            const forcedType = typeToggle?.dataset?.type;
            const categorical = forcedType === 'categorical' ? true :
                                forcedType === 'numeric' ? false :
                                levels.some(l => typeof l === 'string');

            factors.push({
                name,
                levels,
                units,
                categorical,
            });
        }
    });

    if (factors.length === 0) {
        alert('Please add at least one factor with levels.');
        return;
    }

    const data = {
        factors,
        design_type: selectedDesignType,
        replicates: parseInt(document.getElementById('replicates').value),
        center_points: parseInt(document.getElementById('center-points').value),
        resolution: parseInt(document.getElementById('resolution').value),
        seed: document.getElementById('seed').value || undefined,
    };

    // D-optimal / I-optimal: add num_runs and model
    if (selectedDesignType === 'd_optimal' || selectedDesignType === 'i_optimal') {
        const numRuns = parseInt(document.getElementById('optimal-num-runs').value);
        if (!isNaN(numRuns) && numRuns > 0) data.num_runs = numRuns;
        data.model = document.getElementById('optimal-model').value;
    }

    // Taguchi: add array selection
    if (selectedDesignType === 'taguchi') {
        data.taguchi_array = document.getElementById('taguchi-array').value;
    }

    try {
        const response = await fetch('/api/experimenter/design/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (response.ok) {
            currentDesign = result.design;
            currentAnalysis = null;
            saveState();
            displayDesign(result);
            updateResultsPanel();
            updateContextBadge();
            svendTrack('feature_use', {category: 'experimenter', action: selectedDesignType});
            // Wizard: mark configure complete and advance to design step
            completeStep('configure');
            goToStep('design');
        } else {
            alert(result.error || 'Failed to generate design');
        }
    } catch (err) {
        alert('Error: ' + err.message);
    }
}

function displayDesign(result) {
    document.getElementById('design-output-empty').style.display = 'none';
    document.getElementById('design-output').style.display = 'block';

    const design = result.design;
    document.getElementById('design-name').textContent = design.name;

    // Summary
    document.getElementById('design-summary').innerHTML = `
        <p><strong>Type:</strong> ${design.design_type}</p>
        <p><strong>Total Runs:</strong> ${design.properties.num_runs}</p>
        ${design.properties.resolution ? `<p><strong>Resolution:</strong> ${['', 'I', 'II', 'III', 'IV', 'V'][design.properties.resolution] || design.properties.resolution}</p>` : ''}
        ${design.properties.num_center_points ? `<p><strong>Center Points:</strong> ${design.properties.num_center_points}</p>` : ''}
    `;

    // Matrix table
    let tableHtml = '<table class="data-table"><thead><tr><th>Run</th>';
    design.factors.forEach(f => {
        tableHtml += `<th>${f.name}${f.units ? ` (${f.units})` : ''}</th>`;
    });
    tableHtml += '</tr></thead><tbody>';

    design.runs.sort((a, b) => a.run_order - b.run_order).forEach(run => {
        const rowClass = run.center_point ? ' class="center-point-row"' : '';
        tableHtml += `<tr${rowClass}><td class="center run-number">${run.run_order}</td>`;
        design.factors.forEach(f => {
            tableHtml += `<td class="center">${run.levels[f.name]}</td>`;
        });
        tableHtml += '</tr>';
    });
    tableHtml += '</tbody></table>';

    if (design.properties.num_center_points > 0) {
        tableHtml += '<p style="font-size: 0.75rem; color: var(--text-dim);">* Center point</p>';
    }

    document.getElementById('design-matrix-container').innerHTML = tableHtml;

    // Alias structure with resolution badge
    if (result.alias_structure) {
        const aliasBox = document.getElementById('alias-structure');
        aliasBox.style.display = 'block';
        const resNum = result.alias_structure.resolution;
        const resRoman = ['', 'I', 'II', 'III', 'IV', 'V'][resNum] || resNum;
        const badgeClass = resNum <= 3 ? 'res-iii' : resNum === 4 ? 'res-iv' : 'res-v';
        let aliasHtml = `
            <h4>Alias Structure <span class="alias-resolution-badge ${badgeClass}">Resolution ${resRoman}</span></h4>
            <p>${result.alias_structure.interpretation}</p>
        `;
        // Show alias table if available
        if (result.alias_structure.aliases && result.alias_structure.aliases.length > 0) {
            aliasHtml += '<details style="margin-top:0.5rem;"><summary style="cursor:pointer;font-size:0.8rem;color:var(--text-secondary);">Show alias pairs</summary>';
            aliasHtml += '<table class="data-table" style="margin-top:0.5rem;font-size:0.8rem;"><tbody>';
            result.alias_structure.aliases.forEach(a => {
                aliasHtml += `<tr><td>${a}</td></tr>`;
            });
            aliasHtml += '</tbody></table></details>';
        }
        aliasBox.innerHTML = aliasHtml;
    } else {
        document.getElementById('alias-structure').style.display = 'none';
    }

    // Notes
    if (design.notes?.length > 0) {
        document.getElementById('design-summary').innerHTML +=
            '<div style="margin-top: 0.5rem; font-size: 0.8rem;">' +
            design.notes.map(n => `<p style="margin: 0.2rem 0;">- ${n}</p>`).join('') +
            '</div>';
    }
}

function updateResultsPanel() {
    if (!currentDesign) {
        document.getElementById('no-design-message').style.display = 'block';
        document.getElementById('results-entry').style.display = 'none';
        return;
    }

    document.getElementById('no-design-message').style.display = 'none';
    document.getElementById('results-entry').style.display = 'block';

    // Build results table
    let tableHtml = '<table class="data-table"><thead><tr><th>Run</th>';
    currentDesign.factors.forEach(f => {
        tableHtml += `<th>${f.name}</th>`;
    });
    tableHtml += '<th>Response</th></tr></thead><tbody>';

    currentDesign.runs.sort((a, b) => a.run_order - b.run_order).forEach(run => {
        tableHtml += `<tr><td class="center run-number">${run.run_order}</td>`;
        currentDesign.factors.forEach(f => {
            tableHtml += `<td class="center">${run.levels[f.name]}</td>`;
        });
        tableHtml += `<td class="center"><input type="number" step="any" class="response-input" data-run-id="${run.run_id}" onchange="saveResponses()"></td>`;
        tableHtml += '</tr>';
    });
    tableHtml += '</tbody></table>';

    document.getElementById('results-table-container').innerHTML = tableHtml;
}

// Download design
function downloadDesign(format) {
    if (!currentDesign) return;

    let content, filename, type;

    if (format === 'csv') {
        const headers = ['Run', ...currentDesign.factors.map(f => f.name), 'Response'];
        const rows = currentDesign.runs.map(r =>
            [r.run_order, ...currentDesign.factors.map(f => r.levels[f.name]), ''].join(',')
        );
        content = [headers.join(','), ...rows].join('\n');
        filename = 'experiment_design.csv';
        type = 'text/csv';
    } else {
        const headers = ['Run', ...currentDesign.factors.map(f => f.name), 'Response'];
        const rows = currentDesign.runs.map(r =>
            [r.run_order, ...currentDesign.factors.map(f => r.levels[f.name]), ''].join('\t')
        );
        content = [headers.join('\t'), ...rows].join('\n');
        filename = 'experiment_design.xls';
        type = 'application/vnd.ms-excel';
    }

    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// Live preview — updates visualization in the Configure step
function updateLivePreview(k, levelCounts, total, baseRuns, reps, cp) {
    // Update stat cards
    const runsEl = document.getElementById('preview-runs');
    const factorsEl = document.getElementById('preview-factors');
    const dfEl = document.getElementById('preview-df');
    if (!runsEl) return; // No preview panel

    runsEl.textContent = total || '--';
    runsEl.className = 'preview-stat-value ' + (total > 100 ? 'runs-red' : total > 50 ? 'runs-yellow' : 'runs-green');
    factorsEl.textContent = k;

    // Estimate error degrees of freedom
    const numTerms = 1 + k + (k * (k - 1)) / 2; // constant + main + 2FI
    const errorDF = Math.max(0, total - numTerms);
    dfEl.textContent = k > 0 ? errorDF : '--';

    // Generate visualization
    const plotEl = document.getElementById('preview-plot');
    if (!plotEl) return;

    if (k === 0) {
        plotEl.innerHTML = '<div class="empty-state" style="padding:2rem;"><p style="font-size:0.85rem;">Add factors to preview design points</p></div>';
        return;
    }

    // Collect factor names and levels
    const factors = [];
    document.querySelectorAll('.factor-row').forEach(row => {
        const name = row.querySelector('.factor-name').value.trim() || 'Factor';
        const levelsStr = row.querySelector('.factor-levels').value.trim();
        if (levelsStr) {
            const levels = levelsStr.split(',').map(l => {
                const trimmed = l.trim();
                const num = parseFloat(trimmed);
                return isNaN(num) ? trimmed : num;
            }).filter(l => l !== '');
            if (levels.length >= 2) factors.push({ name, levels });
        }
    });

    if (factors.length === 0) return;

    // Build approximate design points (full factorial of first 2-3 factors)
    const textColor = getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim();

    if (factors.length === 1) {
        // 1D: dot plot on a number line
        const pts = factors[0].levels;
        Plotly.react(plotEl, [{
            x: pts,
            y: pts.map(() => 0),
            type: 'scatter',
            mode: 'markers',
            marker: { size: 12, color: 'var(--accent-primary)' },
        }], {
            xaxis: { title: factors[0].name },
            yaxis: { visible: false, range: [-0.5, 0.5] },
            margin: { t: 10, r: 20, b: 40, l: 40 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: textColor, size: 11 },
            height: 250,
        }, { responsive: true, displayModeBar: false });
    } else if (factors.length === 2) {
        // 2D scatter of design points
        const pts = _cartesianProduct(factors[0].levels, factors[1].levels);
        Plotly.react(plotEl, [{
            x: pts.map(p => p[0]),
            y: pts.map(p => p[1]),
            type: 'scatter',
            mode: 'markers',
            marker: { size: 10, color: 'var(--accent-primary)' },
        }], {
            xaxis: { title: factors[0].name },
            yaxis: { title: factors[1].name },
            margin: { t: 10, r: 20, b: 40, l: 55 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: textColor, size: 11 },
            height: 250,
        }, { responsive: true, displayModeBar: false });
    } else if (factors.length === 3) {
        // 3D scatter
        const pts = _cartesianProduct3(factors[0].levels, factors[1].levels, factors[2].levels);
        Plotly.react(plotEl, [{
            x: pts.map(p => p[0]),
            y: pts.map(p => p[1]),
            z: pts.map(p => p[2]),
            type: 'scatter3d',
            mode: 'markers',
            marker: { size: 5, color: 'var(--accent-primary)' },
        }], {
            scene: {
                xaxis: { title: factors[0].name },
                yaxis: { title: factors[1].name },
                zaxis: { title: factors[2].name },
                bgcolor: 'rgba(0,0,0,0)',
            },
            margin: { t: 10, r: 10, b: 10, l: 10 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: textColor, size: 10 },
            height: 250,
        }, { responsive: true, displayModeBar: false });
    } else {
        // 4+ factors: summary card
        plotEl.innerHTML = `
            <div style="padding:1.5rem;text-align:center;">
                <div style="font-size:2rem;font-weight:700;color:var(--accent-primary);">${k}</div>
                <div style="font-size:0.8rem;color:var(--text-dim);margin-bottom:1rem;">Factors</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.8rem;">
                    ${factors.map(f => `
                        <div style="background:var(--bg-tertiary);padding:0.4rem;border-radius:4px;">
                            <div style="font-weight:600;color:var(--text-primary);">${f.name}</div>
                            <div style="color:var(--text-dim);">${f.levels.length} levels</div>
                        </div>
                    `).join('')}
                </div>
                <div style="margin-top:1rem;font-size:0.85rem;color:var(--text-secondary);">
                    ${total} runs &middot; ${errorDF} error DF
                </div>
            </div>
        `;
    }
}

function _cartesianProduct(a, b) {
    const result = [];
    for (const x of a) for (const y of b) result.push([x, y]);
    return result;
}

function _cartesianProduct3(a, b, c) {
    const result = [];
    for (const x of a) for (const y of b) for (const z of c) result.push([x, y, z]);
    return result;
}
