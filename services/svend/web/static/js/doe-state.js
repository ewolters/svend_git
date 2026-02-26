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

    // Update guidance panel if open
    if (contextPanelOpen) updateGuidance();

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

    // Restore context panel state
    try {
        if (sessionStorage.getItem('doe_panel_open') === 'true') {
            toggleContextPanel();
        }
    } catch (e) {}
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

// Context Panel (right sidebar)
let contextPanelOpen = false;

function toggleContextPanel() {
    contextPanelOpen = !contextPanelOpen;
    const panel = document.getElementById('context-panel');
    const page = document.querySelector('.doe-page');
    const toggle = document.getElementById('context-panel-toggle');

    panel.classList.toggle('open', contextPanelOpen);
    page.classList.toggle('panel-open', contextPanelOpen);
    toggle.classList.toggle('hidden', contextPanelOpen);

    try { sessionStorage.setItem('doe_panel_open', contextPanelOpen); } catch (e) {}

    // Update guidance when opening
    if (contextPanelOpen) updateGuidance();
    window.dispatchEvent(new Event('resize'));
}

function showContextTab(tab) {
    document.querySelectorAll('.context-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.context-tab-content').forEach(c => c.classList.remove('active'));

    const tabContent = document.getElementById('context-tab-' + tab);
    if (tabContent) tabContent.classList.add('active');

    const tabBtn = document.querySelector(`.context-tab[onclick*="'${tab}'"]`);
    if (tabBtn) tabBtn.classList.add('active');

    if (tab === 'guidance') updateGuidance();
}

// Step-aware guidance content
const GUIDANCE = {
    configure: {
        title: 'Configure Your Experiment',
        sections: [
            { heading: 'Choosing a Design Type', text: 'Select based on your goal: <strong>Screening</strong> (Plackett-Burman, Fractional Factorial) to identify important factors from many candidates. <strong>Factorial</strong> for studying factor interactions. <strong>RSM</strong> (CCD, Box-Behnken) for optimization with curvature. <strong>Optimal</strong> (D/I-Optimal) for custom run counts.' },
            { heading: 'Defining Factors', text: 'Enter at least 2 factors with their levels. For continuous factors, use numeric values (e.g., 100, 150). For categorical factors, use text labels (e.g., A, B, C). Most designs work best with 2-level continuous factors.' },
        ],
        tip: 'Start with a screening design if you have 5+ factors. You can always follow up with an RSM design on the significant factors.',
    },
    design: {
        title: 'Review Your Design',
        sections: [
            { heading: 'Design Matrix', text: 'Each row is one experimental run. Execute runs in the randomized order shown to minimize bias from time-related effects.' },
            { heading: 'Power Analysis', text: 'Use the Power tab to check if your design has enough runs to detect meaningful effects. Aim for power > 80% for your expected effect size.' },
            { heading: 'Alias Structure', text: 'For fractional factorial designs, aliases show which effects are confounded. Resolution III aliases main effects with 2-factor interactions. Resolution IV keeps main effects clear.' },
        ],
        tip: 'Export the design matrix to CSV for lab use. Mark center point runs (*) for curvature detection.',
    },
    results: {
        title: 'Enter Experimental Results',
        sections: [
            { heading: 'Data Entry', text: 'Enter the measured response for each run. You can also import results from a CSV file with a "Response" column.' },
            { heading: 'Tips', text: 'Double-check entries for outliers or typos. If runs were not completed, leave them blank — the analysis handles missing data.' },
        ],
        tip: 'Name your response variable descriptively (e.g., "Yield (%)" or "Surface Roughness (Ra)") — it will appear in all plots and reports.',
    },
    analyze: {
        title: 'Interpret Your Results',
        sections: [
            { heading: 'ANOVA Table', text: 'Look for terms with p-values below your significance level (alpha). These factors have a statistically significant effect on the response. The "Sig" badge marks significant terms.' },
            { heading: 'R-Squared', text: 'R² shows model fit: <strong>≥90%</strong> (green) is excellent, <strong>≥70%</strong> is good, <strong>≥50%</strong> is moderate. Compare R² and R²(adj) — a large gap suggests overfitting.' },
            { heading: 'Effects Plots', text: 'Main effects show each factor\'s individual impact. Interaction plots reveal factor combinations. The Pareto chart ranks effects by magnitude.' },
            { heading: 'Residuals', text: 'Check the normal probability plot for normality. Residuals vs fitted should show random scatter. Patterns indicate model inadequacy.' },
        ],
        tip: 'Use the alpha slider to explore significance at different levels. Marginal effects (p between alpha and 2×alpha) may be worth investigating.',
    },
    optimize: {
        title: 'Optimize Your Response',
        sections: [
            { heading: 'Contour Plot', text: 'Visualize how two factors jointly affect the response. Toggle to 3D Surface for a perspective view. Use hold sliders to explore additional factor dimensions.' },
            { heading: 'Response Optimizer', text: 'Set goals (maximize, minimize, or target) and bounds for each response. The optimizer finds factor settings that best satisfy all goals simultaneously using desirability functions.' },
            { heading: 'Multi-Response', text: 'Click "+ Add Response" for trade-off optimization. Adjust weights to prioritize one response over another. The composite desirability balances all goals.' },
        ],
        tip: 'For RSM designs, always check contour plots for saddle points before trusting the optimizer. A saddle point means the optimal is at the boundary.',
    },
};

function updateGuidance() {
    const el = document.getElementById('guidance-content');
    if (!el) return;

    const guide = GUIDANCE[currentStep];
    if (!guide) return;

    let html = `<h3 style="font-size:1rem;margin:0 0 1rem;color:var(--text-primary);">${guide.title}</h3>`;

    guide.sections.forEach(s => {
        html += `<div class="guidance-section"><h4>${s.heading}</h4><p>${s.text}</p></div>`;
    });

    if (guide.tip) {
        html += `<div class="guidance-tip"><strong>Tip:</strong> ${guide.tip}</div>`;
    }

    el.innerHTML = html;
}
