// =============================================================================
// Init
// =============================================================================

(async function init() {
    // Check if loading existing simulation from URL
    const match = window.location.pathname.match(/simulator\/([0-9a-f-]+)\//);
    if (match) {
        try {
            const resp = await fetch(`/api/plantsim/${match[1]}/`, { headers: { 'X-CSRFToken': getCsrf() } });
            if (resp.ok) {
                const data = await resp.json();
                if (data.simulation) loadSimulation(data.simulation);
                return;
            }
        } catch (e) { /* fall through to empty canvas */ }
    }
    renderCanvas();
    resetView();
})();
