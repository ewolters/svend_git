// DOE Workbench — State Management
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
let currentComponent = 'design';
let currentSubTab = 'create';

// Persist DOE state across page refreshes
function saveState() {
    try {
        if (currentDesign) sessionStorage.setItem('doe_design', JSON.stringify(currentDesign));
        else sessionStorage.removeItem('doe_design');
        if (currentAnalysis) sessionStorage.setItem('doe_analysis', JSON.stringify(currentAnalysis));
        else sessionStorage.removeItem('doe_analysis');
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
});

// Component Navigation
function showComponent(id) {
    document.querySelectorAll('.component-content').forEach(c => c.classList.remove('active'));
    document.querySelectorAll('.component-btn').forEach(b => b.classList.remove('active'));

    document.getElementById(id + '-component').classList.add('active');
    document.querySelector(`.component-btn[onclick="showComponent('${id}')"]`).classList.add('active');
    currentComponent = id;
}

function showSubTab(component, tab) {
    const container = document.getElementById(component + '-component');
    container.querySelectorAll('.sub-content').forEach(c => c.classList.remove('active'));
    container.querySelectorAll('.sub-tab').forEach(t => t.classList.remove('active'));

    document.getElementById(component + '-' + tab).classList.add('active');
    // Activate the matching sub-tab button (find by onclick text or fallback)
    const tabBtn = container.querySelector(`.sub-tab[onclick*="'${tab}'"]`);
    if (tabBtn) tabBtn.classList.add('active');
    currentSubTab = tab;
}
