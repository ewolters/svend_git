// ============================================================================
// Init
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // ========================================
    // Restore auto-saved state FIRST
    // ========================================
    const autoSaved = localStorage.getItem(AUTO_SAVE_KEY);
    if (autoSaved) {
        try {
            restoreState(JSON.parse(autoSaved));
        } catch (e) {
            console.warn('Failed to restore auto-save:', e);
        }
    }

    // ========================================
    // Initialize custom steppers
    // ========================================
    initializeSteppers();

    // ========================================
    // Initialize all calculators
    // ========================================
    calcTakt();
    calcRTO();
    calcKanban();
    calcEPEI();
    calcSafety();
    calcEOQ();
    calcOEE();
    renderYamazumiInputs();
    renderBottleneckInputs();
    calcLittles();
    calcQueue();
    calcPitch();
    renderRTYInputs();
    calcDPMO();
    calcTurns();
    calcCOQ();
    renderSMEDInputs();
    renderChangeoverMatrix();
    renderFMEAInputs();
    calcCpk();
    calcSampleSize();
    calcLineEff();
    calcOLE();
    renderCycleInputs();
    renderBAInputs();
    renderHeijunkaInputs();
    // Queuing Lab
    calcQueueFinite();
    renderPriorityClasses();
    calcOptimizer();
    initSimVisual();
    initCompareVisuals();
    renderTandemStages();
    // Line Simulator
    initLineSim();
    // Scheduling Tools
    renderSequencerJobs();
    renderCapacityOrders();
    renderMixedProducts();
    renderDDSOrders();
    // Method Tools
    renderQFDWhats();
    renderQFDHows();
    // Flow Analysis
    renderPFASteps();
    renderWFAElements();
    // Advanced Tools
    calcMTBF();
    calcErlang();
    riskRenderTable();
    riskUpdate();
    // SIOP Tools
    loadSampleABCItems();
    loadSampleDPPeriods();
    calcServiceLevel();
    loadSampleMRPPeriods();

    // Guide widget for default calculator
    renderGuide('takt');

    // ========================================
    // Persistence: scenarios & auto-save
    // ========================================
    updateScenarioDropdown();
    hookAutoSave();
});

// Cell Design Simulator — see calc-sim-flow.js


// Kanban Simulator — see calc-sim-flow.js

// Beer Game — see calc-sim-flow.js

// TOC / Drum-Buffer-Rope — see calc-sim-flow.js

// Advanced Calculators (Desirability, SPC Rare Events, Probit, MTBF, Erlang, Risk Matrix) — see calc-advanced.js
// Process Bayesian Statistics (Cpk, Belief Chart, Evidence Strength, Sigma) — see calc-pbs.js

// ============================================================================
// FMEA Monte Carlo — Risk Distribution Simulation
// ============================================================================

// FMEA Monte Carlo — see calc-sim-quality.js

// SMED Simulator — see calc-sim-quality.js

// Heijunka Simulator — see calc-sim-quality.js

