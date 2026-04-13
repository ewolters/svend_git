// Core Infrastructure (calcMeta, SvendOps, Guide, Stepper, MonteCarlo, showCalc) — see calc-core.js
// VSM Import/Export — see calc-vsm.js
// Production Calculators (Takt, RTO, Yamazumi, Kanban, EPEI, Safety, EOQ, OEE, Bottleneck, Littles) — see calc-production.js
// Queue Theory Calculators — see calc-queue.js
// Production Calculators (Pitch) — see calc-production.js
// Quality Calculators (RTY, DPMO) — see calc-quality.js
// Production Calculators (Inventory Turns, Cost of Quality) — see calc-production.js
// Lean Calculators (SMED, Changeover Matrix, PFA, WFA) — see calc-lean.js
// Quality Calculators (FMEA, Cpk, Sample Size, Power Analysis) — see calc-quality.js
// Production Calculators (LineEff, OLE, Cycle Time, Before/After, Heijunka) — see calc-production.js
// Scheduling Tools (Job Sequencer, Sequence Optimizer, Capacity Load, Mixed-Model, Due Date Risk) — see calc-scheduling.js
// House of Quality (QFD) — see calc-qfd.js

// ============================================================================
// DSW Integration (Stub)
// ============================================================================

function pullDSWStats(type) {
    alert('DSW integration coming soon!\n\nThis will let you pull statistics directly from time studies and descriptive analyses.');
}

// ============================================================================
// Scenario Persistence (LocalStorage)
// ============================================================================

const STORAGE_KEY = 'svend_ops_scenarios';
const AUTO_SAVE_KEY = 'svend_ops_autosave';
let currentScenarioId = null;

// Collect all calculator state into one object
function collectState() {
    // Collect all calculator input values generically
    const calcInputs = {};
    document.querySelectorAll('.calc-layout input, .calc-layout select').forEach(el => {
        if (el.id) calcInputs[el.id] = el.type === 'checkbox' ? el.checked : el.value;
    });

    return {
        // Active calculator
        currentCalcId: currentCalcId,

        // All calculator input values
        calcInputs: calcInputs,

        // Line Simulator
        lineStations: lineStations,
        lineProducts: lineProducts,
        lineOrders: lineOrders,
        lineSimSettings: {
            cov: parseFloat(document.getElementById('ls-cov')?.value) || 0.1,
            maxWip: parseInt(document.getElementById('ls-max-wip')?.value) || 3,
            batchSize: parseInt(document.getElementById('ls-batch-size')?.value) || 1,
            changeoverTime: parseFloat(document.getElementById('ls-changeover-time')?.value) || 120,
            simMode: document.getElementById('ls-sim-mode')?.value || 'infinite'
        },

        // SMED
        smedData: smedData,
        smedBaseline: smedBaseline,
        smedImpact: {
            changoversDay: parseFloat(document.getElementById('smed-changeovers-day')?.value) || 4,
            daysYear: parseFloat(document.getElementById('smed-days-year')?.value) || 250,
            hourlyCost: parseFloat(document.getElementById('smed-hourly-cost')?.value) || 150
        },

        // Yamazumi
        yamazumiData: typeof yamazumiData !== 'undefined' ? yamazumiData : [],
        yamazumiTakt: parseFloat(document.getElementById('yamazumi-takt')?.value) || 60,

        // Changeover Matrix
        changeoverMatrix: changeoverMatrix,
        changeoverProducts: document.getElementById('changeover-products')?.value || 'A, B, C, D',

        // Bottleneck
        bottleneckData: typeof bottleneckData !== 'undefined' ? bottleneckData : [],

        // FMEA
        fmeaData: typeof fmeaData !== 'undefined' ? fmeaData : [],

        // RTY
        rtyData: typeof rtyData !== 'undefined' ? rtyData : [],

        // Cycle Time Study
        cycleData: typeof cycleData !== 'undefined' ? cycleData : [],

        // Before/After
        baData: typeof baData !== 'undefined' ? baData : [],

        // Heijunka
        heijunkaData: typeof heijunkaData !== 'undefined' ? heijunkaData : [],

        // Multi-Stage Queue
        tandemStages: typeof tandemStages !== 'undefined' ? tandemStages : [],

        // Priority Queue
        priorityClasses: typeof priorityClasses !== 'undefined' ? priorityClasses : [],

        // Scheduling Tools
        sequencerJobs: typeof sequencerJobs !== 'undefined' ? sequencerJobs : [],
        sequencerOrder: typeof sequencerOrder !== 'undefined' ? sequencerOrder : [],
        capacityOrders: typeof capacityOrders !== 'undefined' ? capacityOrders : [],
        mixedProducts: typeof mixedProducts !== 'undefined' ? mixedProducts : [],
        ddsOrders: typeof ddsOrders !== 'undefined' ? ddsOrders : [],

        // QFD
        qfdWhats: typeof qfdWhats !== 'undefined' ? qfdWhats : [],
        qfdHows: typeof qfdHows !== 'undefined' ? qfdHows : [],
        qfdRelationships: typeof qfdRelationships !== 'undefined' ? qfdRelationships : {},
        qfdCorrelations: typeof qfdCorrelations !== 'undefined' ? qfdCorrelations : {},
        qfdWhatCorrelations: typeof qfdWhatCorrelations !== 'undefined' ? qfdWhatCorrelations : {},
        qfdParts: typeof qfdParts !== 'undefined' ? qfdParts : [],
        qfdProcesses: typeof qfdProcesses !== 'undefined' ? qfdProcesses : [],
        qfdControls: typeof qfdControls !== 'undefined' ? qfdControls : [],
        qfdMatrix2: typeof qfdMatrix2 !== 'undefined' ? qfdMatrix2 : {},
        qfdMatrix3: typeof qfdMatrix3 !== 'undefined' ? qfdMatrix3 : {},
        qfdMatrix4: typeof qfdMatrix4 !== 'undefined' ? qfdMatrix4 : {},

        // PFA — Product Flow Analysis
        pfaSteps: typeof pfaSteps !== 'undefined' ? pfaSteps : [],
        pfaBaseline: typeof pfaBaseline !== 'undefined' ? pfaBaseline : null,

        // WFA — Workflow Analysis
        wfaElements: typeof wfaElements !== 'undefined' ? wfaElements : [],
        wfaBaseline: typeof wfaBaseline !== 'undefined' ? wfaBaseline : null,

        // Timestamp
        savedAt: new Date().toISOString()
    };
}

// Restore state from saved object
function restoreState(state) {
    if (!state) return;

    // Restore all calculator inputs generically
    if (state.calcInputs) {
        for (const [id, val] of Object.entries(state.calcInputs)) {
            const el = document.getElementById(id);
            if (!el) continue;
            if (el.type === 'checkbox') el.checked = val;
            else el.value = val;
        }
    }

    // Restore active calculator
    if (state.currentCalcId && typeof showCalc === 'function') {
        showCalc(state.currentCalcId);
    }

    // Line Simulator
    if (state.lineStations) lineStations = state.lineStations;
    if (state.lineProducts) lineProducts = state.lineProducts;
    if (state.lineOrders) lineOrders = state.lineOrders;
    if (state.lineSimSettings) {
        const s = state.lineSimSettings;
        if (document.getElementById('ls-cov')) document.getElementById('ls-cov').value = s.cov;
        if (document.getElementById('ls-max-wip')) document.getElementById('ls-max-wip').value = s.maxWip;
        if (document.getElementById('ls-batch-size')) document.getElementById('ls-batch-size').value = s.batchSize;
        if (document.getElementById('ls-changeover-time')) document.getElementById('ls-changeover-time').value = s.changeoverTime;
        if (document.getElementById('ls-sim-mode')) document.getElementById('ls-sim-mode').value = s.simMode;
    }

    // SMED
    if (state.smedData) smedData = state.smedData;
    if (state.smedBaseline) {
        smedBaseline = state.smedBaseline;
        document.getElementById('smed-clear-baseline').style.display = 'inline-block';
    }
    if (state.smedImpact) {
        if (document.getElementById('smed-changeovers-day')) document.getElementById('smed-changeovers-day').value = state.smedImpact.changoversDay;
        if (document.getElementById('smed-days-year')) document.getElementById('smed-days-year').value = state.smedImpact.daysYear;
        if (document.getElementById('smed-hourly-cost')) document.getElementById('smed-hourly-cost').value = state.smedImpact.hourlyCost;
    }

    // Yamazumi
    if (state.yamazumiData && typeof yamazumiData !== 'undefined') yamazumiData = state.yamazumiData;
    if (state.yamazumiTakt && document.getElementById('yamazumi-takt')) {
        document.getElementById('yamazumi-takt').value = state.yamazumiTakt;
    }

    // Changeover Matrix
    if (state.changeoverMatrix) changeoverMatrix = state.changeoverMatrix;
    if (state.changeoverProducts && document.getElementById('changeover-products')) {
        document.getElementById('changeover-products').value = state.changeoverProducts;
    }

    // Bottleneck
    if (state.bottleneckData && typeof bottleneckData !== 'undefined') bottleneckData = state.bottleneckData;

    // FMEA
    if (state.fmeaData && typeof fmeaData !== 'undefined') fmeaData = state.fmeaData;

    // RTY
    if (state.rtyData && typeof rtyData !== 'undefined') rtyData = state.rtyData;

    // Cycle Time Study
    if (state.cycleData && typeof cycleData !== 'undefined') cycleData = state.cycleData;

    // Before/After
    if (state.baData && typeof baData !== 'undefined') baData = state.baData;

    // Heijunka
    if (state.heijunkaData && typeof heijunkaData !== 'undefined') heijunkaData = state.heijunkaData;

    // Multi-Stage Queue
    if (state.tandemStages && typeof tandemStages !== 'undefined') tandemStages = state.tandemStages;

    // Priority Queue
    if (state.priorityClasses && typeof priorityClasses !== 'undefined') priorityClasses = state.priorityClasses;

    // Scheduling Tools
    if (state.sequencerJobs && typeof sequencerJobs !== 'undefined') sequencerJobs = state.sequencerJobs;
    if (state.sequencerOrder && typeof sequencerOrder !== 'undefined') sequencerOrder = state.sequencerOrder;
    if (state.capacityOrders && typeof capacityOrders !== 'undefined') capacityOrders = state.capacityOrders;
    if (state.mixedProducts && typeof mixedProducts !== 'undefined') mixedProducts = state.mixedProducts;
    if (state.ddsOrders && typeof ddsOrders !== 'undefined') ddsOrders = state.ddsOrders;

    // QFD
    if (state.qfdWhats && typeof qfdWhats !== 'undefined') qfdWhats = state.qfdWhats;
    if (state.qfdHows && typeof qfdHows !== 'undefined') qfdHows = state.qfdHows;
    if (state.qfdRelationships && typeof qfdRelationships !== 'undefined') qfdRelationships = state.qfdRelationships;
    if (state.qfdCorrelations && typeof qfdCorrelations !== 'undefined') qfdCorrelations = state.qfdCorrelations;
    if (state.qfdWhatCorrelations && typeof qfdWhatCorrelations !== 'undefined') qfdWhatCorrelations = state.qfdWhatCorrelations;
    if (state.qfdParts && typeof qfdParts !== 'undefined') qfdParts = state.qfdParts;
    if (state.qfdProcesses && typeof qfdProcesses !== 'undefined') qfdProcesses = state.qfdProcesses;
    if (state.qfdControls && typeof qfdControls !== 'undefined') qfdControls = state.qfdControls;
    if (state.qfdMatrix2 && typeof qfdMatrix2 !== 'undefined') qfdMatrix2 = state.qfdMatrix2;
    if (state.qfdMatrix3 && typeof qfdMatrix3 !== 'undefined') qfdMatrix3 = state.qfdMatrix3;
    if (state.qfdMatrix4 && typeof qfdMatrix4 !== 'undefined') qfdMatrix4 = state.qfdMatrix4;

    // PFA — Product Flow Analysis
    if (state.pfaSteps && typeof pfaSteps !== 'undefined') pfaSteps = state.pfaSteps;
    if (state.pfaBaseline && typeof pfaBaseline !== 'undefined') {
        pfaBaseline = state.pfaBaseline;
        const btn = document.getElementById('pfa-clear-baseline');
        if (btn) btn.style.display = 'inline-block';
    }

    // WFA — Workflow Analysis
    if (state.wfaElements && typeof wfaElements !== 'undefined') wfaElements = state.wfaElements;
    if (state.wfaBaseline && typeof wfaBaseline !== 'undefined') {
        wfaBaseline = state.wfaBaseline;
        const btn = document.getElementById('wfa-clear-baseline');
        if (btn) btn.style.display = 'inline-block';
    }
}

// Auto-save current state (debounced)
let autoSaveTimeout = null;
function autoSave() {
    clearTimeout(autoSaveTimeout);
    autoSaveTimeout = setTimeout(() => {
        const state = collectState();
        localStorage.setItem(AUTO_SAVE_KEY, JSON.stringify(state));
    }, 1000); // 1 second debounce
}

// Get all saved scenarios
function getScenarios() {
    const data = localStorage.getItem(STORAGE_KEY);
    return data ? JSON.parse(data) : {};
}

// Save scenarios to localStorage
function setScenarios(scenarios) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(scenarios));
}

// Update scenario dropdown
function updateScenarioDropdown() {
    const select = document.getElementById('scenario-select');
    if (!select) return;

    const scenarios = getScenarios();
    const scenarioList = Object.entries(scenarios).sort((a, b) =>
        new Date(b[1].savedAt) - new Date(a[1].savedAt)
    );

    select.innerHTML = '<option value="">Current Session</option>';
    scenarioList.forEach(([id, scenario]) => {
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = scenario.name || `Scenario ${id.slice(0, 6)}`;
        if (id === currentScenarioId) opt.selected = true;
        select.appendChild(opt);
    });

    // Handle selection change
    select.onchange = () => {
        const id = select.value;
        if (id) {
            loadScenario(id);
        } else {
            currentScenarioId = null;
        }
    };
}

// Save current state as scenario
function saveScenario() {
    const state = collectState();

    if (currentScenarioId) {
        // Update existing scenario
        const scenarios = getScenarios();
        if (scenarios[currentScenarioId]) {
            state.name = scenarios[currentScenarioId].name;
            scenarios[currentScenarioId] = state;
            setScenarios(scenarios);
            showToast(`Saved "${state.name}"`);
            return;
        }
    }

    // No current scenario - prompt to save as new
    saveScenarioAs();
}

// Save as new scenario with name
async function saveScenarioAs() {
    const name = await svendPrompt('Scenario name:', `Scenario ${new Date().toLocaleDateString()}`);
    if (!name) return;

    const state = collectState();
    state.name = name;

    const id = Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
    const scenarios = getScenarios();
    scenarios[id] = state;
    setScenarios(scenarios);

    currentScenarioId = id;
    updateScenarioDropdown();
    toggleScenarioMenu();
    showToast(`Saved "${name}"`);
}

// Load a scenario by ID
function loadScenario(id) {
    const scenarios = getScenarios();
    const scenario = scenarios[id];
    if (!scenario) return;

    restoreState(scenario);
    currentScenarioId = id;

    // Refresh all displays
    refreshAllDisplays();
    showToast(`Loaded "${scenario.name}"`);
}

// Rename current scenario
async function renameScenario() {
    if (!currentScenarioId) {
        alert('No scenario selected. Save first.');
        return;
    }

    const scenarios = getScenarios();
    const current = scenarios[currentScenarioId];
    if (!current) return;

    const name = await svendPrompt('New name:', current.name);
    if (!name) return;

    current.name = name;
    setScenarios(scenarios);
    updateScenarioDropdown();
    toggleScenarioMenu();
    showToast(`Renamed to "${name}"`);
}

// Delete current scenario
function deleteScenario() {
    if (!currentScenarioId) {
        alert('No scenario selected.');
        return;
    }

    const scenarios = getScenarios();
    const name = scenarios[currentScenarioId]?.name || 'this scenario';

    if (!confirm(`Delete "${name}"? This cannot be undone.`)) return;

    delete scenarios[currentScenarioId];
    setScenarios(scenarios);
    currentScenarioId = null;
    updateScenarioDropdown();
    toggleScenarioMenu();
    showToast('Scenario deleted');
}

// Export all scenarios to JSON file
function exportScenarios() {
    const scenarios = getScenarios();
    const autoSave = localStorage.getItem(AUTO_SAVE_KEY);

    const exportData = {
        scenarios: scenarios,
        currentSession: autoSave ? JSON.parse(autoSave) : collectState(),
        exportedAt: new Date().toISOString(),
        version: 1
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `svend-ops-scenarios-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);

    toggleScenarioMenu();
    showToast('Exported all scenarios');
}

// Import scenarios from JSON file
function importScenarios(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const data = JSON.parse(e.target.result);

            if (data.scenarios) {
                const existing = getScenarios();
                const merged = { ...existing, ...data.scenarios };
                setScenarios(merged);
                updateScenarioDropdown();

                const count = Object.keys(data.scenarios).length;
                showToast(`Imported ${count} scenario${count !== 1 ? 's' : ''}`);
            }

            // Optionally restore current session
            if (data.currentSession) {
                if (confirm('Also restore the session state from this export?')) {
                    restoreState(data.currentSession);
                    refreshAllDisplays();
                }
            }
        } catch (err) {
            alert('Invalid file format: ' + safeStr(err, 'Unknown error'));
        }
    };
    reader.readAsText(file);
    event.target.value = ''; // Reset file input
    toggleScenarioMenu();
}

// Toggle scenario menu dropdown
function toggleScenarioMenu() {
    const menu = document.getElementById('scenario-menu');
    menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
}

// Close menu when clicking outside
document.addEventListener('click', (e) => {
    const menu = document.getElementById('scenario-menu');
    const toggle = e.target.closest('[onclick*="toggleScenarioMenu"]');
    if (menu && !menu.contains(e.target) && !toggle) {
        menu.style.display = 'none';
    }
});

// Refresh all calculator displays after state restore
function refreshAllDisplays() {
    // Line Simulator
    if (typeof renderLineSimInputs === 'function') renderLineSimInputs();
    if (typeof updateSimMode === 'function') updateSimMode();
    if (typeof renderProducts === 'function') renderProducts();
    if (typeof renderOrders === 'function') renderOrders();
    if (typeof initLineSim === 'function') initLineSim();

    // SMED
    if (typeof renderSMEDInputs === 'function') renderSMEDInputs();

    // Yamazumi
    if (typeof renderYamazumiInputs === 'function') renderYamazumiInputs();

    // Changeover Matrix
    if (typeof renderChangeoverMatrix === 'function') renderChangeoverMatrix();

    // Bottleneck
    if (typeof renderBottleneckInputs === 'function') renderBottleneckInputs();

    // FMEA
    if (typeof renderFMEAInputs === 'function') renderFMEAInputs();

    // RTY
    if (typeof renderRTYInputs === 'function') renderRTYInputs();

    // Cycle Time
    if (typeof renderCycleInputs === 'function') renderCycleInputs();

    // Before/After
    if (typeof renderBAInputs === 'function') renderBAInputs();

    // Heijunka
    if (typeof renderHeijunkaInputs === 'function') renderHeijunkaInputs();

    // Multi-Stage Queue
    if (typeof renderTandemStages === 'function') renderTandemStages();

    // Priority Queue
    if (typeof renderPriorityClasses === 'function') renderPriorityClasses();

    // Scheduling Tools
    if (typeof renderSequencerJobs === 'function') renderSequencerJobs();
    if (typeof renderCapacityOrders === 'function') renderCapacityOrders();
    if (typeof renderMixedProducts === 'function') renderMixedProducts();
    if (typeof renderDDSOrders === 'function') renderDDSOrders();

    // QFD
    if (typeof renderQFDWhats === 'function') renderQFDWhats();
    if (typeof renderQFDHows === 'function') renderQFDHows();

    // PFA — Product Flow Analysis
    if (typeof renderPFASteps === 'function') renderPFASteps();

    // WFA — Workflow Analysis
    if (typeof renderWFAElements === 'function') renderWFAElements();

    // Recalculate all simple calculators from restored inputs
    if (typeof calcTakt === 'function') calcTakt();
    if (typeof calcRTO === 'function') calcRTO();
    if (typeof calcOEE === 'function') calcOEE();
    if (typeof calcKanban === 'function') calcKanban();
    if (typeof calcEPEI === 'function') calcEPEI();
    if (typeof calcSafety === 'function') calcSafety();
    if (typeof calcEOQ === 'function') calcEOQ();
    if (typeof calcLittles === 'function') calcLittles();
    if (typeof calcQueue === 'function') calcQueue();
    if (typeof calcPitch === 'function') calcPitch();
    if (typeof calcBottleneck === 'function') calcBottleneck();
    if (typeof calcMTBF === 'function') calcMTBF();
    if (typeof renderYamazumi === 'function') renderYamazumi();
}

// Toast notification helper
function showToast(message) {
    svendToast(message);
}

// Hook auto-save into state-changing functions
function hookAutoSave() {
    // Override update functions to trigger auto-save
    const originalFunctions = [
        'updateSMED', 'addSMEDElement', 'removeSMED',
        'addYamazumiStation', 'removeYamazumiStation', 'updateYamazumiStation',
        'updateMatrixValue',
        'updateBottleneck', 'addBottleneckStation',
        'updateFMEA', 'addFMEARow',
        'updateRTY', 'addRTYStep',
        'updateCycle', 'addCycleMeasurement',
        'updateBA', 'addBAMetric',
        'updateHeijunka', 'addHeijunkaProduct',
        'updateTandemStage', 'addTandemStage',
        'updatePriorityClass', 'addPriorityClass',
        'addLineStation', 'removeLineStation', 'updateLineStation',
        'addProduct', 'removeProduct',
        'addOrder', 'removeOrder',
        // Scheduling
        'addSequencerJob', 'removeSequencerJob', 'updateSequencerJob',
        'addCapacityOrder', 'removeCapacityOrder', 'updateCapacityOrder',
        'addMixedProduct', 'removeMixedProduct', 'updateMixedProduct',
        'addDDSOrder', 'removeDDSOrder', 'updateDDSOrder',
        // PFA & WFA
        'addPFAStep', 'removePFAStep', 'updatePFAStep', 'capturePFABaseline', 'clearPFABaseline',
        'addWFAElement', 'removeWFAElement', 'updateWFAElement', 'captureWFABaseline', 'clearWFABaseline'
    ];

    originalFunctions.forEach(fnName => {
        if (typeof window[fnName] === 'function') {
            const original = window[fnName];
            window[fnName] = function(...args) {
                const result = original.apply(this, args);
                autoSave();
                return result;
            };
        }
    });
}

