// DOE Workbench — State Management & Wizard Navigation
// Load order: doe-state.js → doe-design.js → doe-analysis.js → doe-optimize.js → doe-power.js → doe-chat.js

// State
let currentDesign = null;
let currentAnalysis = null;
let designTypes = [];
let selectedDesignType = 'full_factorial';
let projects = [];
let chatHistory = [];
let chatModalOpen = false;
let selectedModel = 'claude';

// Wizard state
let currentStep = 'configure';
let completedSteps = new Set();
const STEP_ORDER = ['configure', 'design', 'results', 'analyze', 'optimize'];

// Step gating rules — returns true if step is accessible
function isStepAccessible(stepId) {
    const idx = STEP_ORDER.indexOf(stepId);
    if (idx <= 0) return true; // Configure always accessible
    if (completedSteps.has(stepId)) return true; // Completed steps stay accessible
    // Check prerequisite
    switch (stepId) {
        case 'design': return !!currentDesign;
        case 'results': return !!currentDesign;
        case 'analyze': return !!currentAnalysis;
        case 'optimize': return !!currentAnalysis;
        default: return true;
    }
}

// Navigate to wizard step
function goToStep(stepId) {
    if (!STEP_ORDER.includes(stepId)) return;
    if (!isStepAccessible(stepId)) return;

    currentStep = stepId;

    // Hide all steps, show target
    document.querySelectorAll('.wizard-step').forEach(s => s.classList.remove('active'));
    const target = document.getElementById('step-' + stepId);
    if (target) target.classList.add('active');

    // Update progress bar
    updateWizardProgress();

    // Persist
    try { sessionStorage.setItem('doe_step', stepId); } catch (e) {}

    // Trigger resize for any Plotly charts in the newly visible step
    window.dispatchEvent(new Event('resize'));
}

// Mark a step as completed and update progress bar
function completeStep(stepId) {
    completedSteps.add(stepId);
    try {
        sessionStorage.setItem('doe_completed_steps', JSON.stringify([...completedSteps]));
    } catch (e) {}
    updateWizardProgress();
}

// Update the visual state of the wizard progress bar
function updateWizardProgress() {
    STEP_ORDER.forEach(stepId => {
        const indicator = document.querySelector(`.wizard-step-indicator[data-step="${stepId}"]`);
        if (!indicator) return;

        indicator.classList.remove('active', 'completed', 'locked');

        if (stepId === currentStep) {
            indicator.classList.add('active');
        } else if (completedSteps.has(stepId)) {
            indicator.classList.add('completed');
        } else if (!isStepAccessible(stepId)) {
            indicator.classList.add('locked');
        }
    });

    // Update connectors
    document.querySelectorAll('.wizard-connector').forEach(conn => {
        const afterStep = conn.dataset.after;
        conn.classList.toggle('completed', completedSteps.has(afterStep));
    });
}

// Persist DOE state across page refreshes
function saveState() {
    try {
        if (currentDesign) sessionStorage.setItem('doe_design', JSON.stringify(currentDesign));
        else sessionStorage.removeItem('doe_design');
        if (currentAnalysis) sessionStorage.setItem('doe_analysis', JSON.stringify(currentAnalysis));
        else sessionStorage.removeItem('doe_analysis');
        sessionStorage.setItem('doe_step', currentStep);
        sessionStorage.setItem('doe_completed_steps', JSON.stringify([...completedSteps]));
    } catch (e) { /* storage full or unavailable */ }
}

function saveResponses() {
    try {
        const vals = {};
        document.querySelectorAll('.response-input').forEach(input => {
            if (input.value) vals[input.dataset.runId] = input.value;
        });
        if (Object.keys(vals).length > 0) sessionStorage.setItem('doe_responses', JSON.stringify(vals));
    } catch (e) { /* ignore */ }
}

function restoreState() {
    try {
        // Restore completed steps
        const cs = sessionStorage.getItem('doe_completed_steps');
        if (cs) completedSteps = new Set(JSON.parse(cs));

        const d = sessionStorage.getItem('doe_design');
        const a = sessionStorage.getItem('doe_analysis');
        if (d) {
            currentDesign = JSON.parse(d);
            displayDesign({ design: currentDesign });
            updateResultsPanel();
            // Restore entered response values
            const r = sessionStorage.getItem('doe_responses');
            if (r) {
                const vals = JSON.parse(r);
                document.querySelectorAll('.response-input').forEach(input => {
                    if (vals[input.dataset.runId]) input.value = vals[input.dataset.runId];
                });
            }
        }
        if (a) {
            currentAnalysis = JSON.parse(a);
            displayAnalysis(currentAnalysis);
        }

        // Restore step
        const step = sessionStorage.getItem('doe_step');
        if (step && STEP_ORDER.includes(step) && isStepAccessible(step)) {
            goToStep(step);
        } else {
            updateWizardProgress();
        }
    } catch (e) { /* corrupt data, ignore */ }
}

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadDesignTypes();
    await loadProjects();
    await loadAvailableModels();
    addFactor();
    addFactor();

    // Restore previous session state (design + analysis)
    restoreState();

    // Check URL params
    const params = new URLSearchParams(window.location.search);
    const projectId = params.get('project');
    if (projectId) {
        document.getElementById('project-selector').value = projectId;
    }

    updateContextBadge();
    updateWizardProgress();
});

// Backward-compat shim — old code may still call showComponent
function showComponent(id) {
    const mapping = { design: 'configure', analysis: 'analyze', optimize: 'optimize' };
    goToStep(mapping[id] || id);
}

// Sub-tab navigation (within Analyze and Optimize steps)
function showSubTab(containerId, tab) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.querySelectorAll('.sub-content').forEach(c => c.classList.remove('active'));
    container.querySelectorAll('.sub-tab').forEach(t => t.classList.remove('active'));

    const target = document.getElementById(containerId + '-' + tab);
    if (target) target.classList.add('active');

    const tabBtn = container.querySelector(`.sub-tab[onclick*="'${tab}'"]`);
    if (tabBtn) tabBtn.classList.add('active');

    window.dispatchEvent(new Event('resize'));
}
