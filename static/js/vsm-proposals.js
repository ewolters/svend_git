/**
 * VSM Proposals Module
 * Kaizen, hypotheses, CI proposals, Hoshin integration, timeline.
 * MIGRATION: Extracted from templates/vsm.html
 */


async function approveProposal(burstId, title, savingsTarget, calcMethod, projectType) {
    try {
        const response = await fetch(`/api/vsm/${vsmId}/approve-proposal/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({
                burst_id: burstId,
                title: title,
                annual_savings_target: savingsTarget,
                calculation_method: calcMethod,
                project_type: projectType,
            }),
        });
        const data = await response.json();
        if (response.status === 409) {
            alert('This proposal has already been approved.');
            return;
        }
        if (!response.ok) {
            alert(safeStr(data.error, 'Failed to approve proposal.'));
            return;
        }
        closeProposalModal();
        alert(`Hoshin project created: ${data.title}\nSavings target: $${data.annual_savings_target.toLocaleString()}/yr\n\nView in Hoshin Kanri.`);
    } catch (err) {
        alert('Error: ' + safeStr(err, 'Unknown error'));
    }
}

// =============================================================================
// View Controls
// =============================================================================

function closeProposalModal() {
    document.getElementById('proposals-modal').style.display = 'none';
}


async function compareStates() {
    if (!vsmId) return;

    try {
        const response = await fetch(`/api/vsm/${vsmId}/compare/`, { credentials: 'include' });
        const data = await response.json();

        if (data.comparison) {
            alert(`Comparison:\nLead Time: ${data.comparison.lead_time.improvement.toFixed(1)}% improvement\nPCE: ${data.comparison.pce.improvement.toFixed(1)}% improvement`);
        } else {
            alert('No future state found to compare.');
        }
    } catch (err) {
        console.error('Compare error:', err);
    }
}


async function createFutureState() {
    if (!vsmId) return;

    try {
        const response = await fetch(`/api/vsm/${vsmId}/future-state/`, {
            method: 'POST',
            credentials: 'include'
        });

        if (!response.ok) throw new Error('Failed to create future state');
        const data = await response.json();
        window.location.href = `/app/vsm/${data.future_state.id}/`;
    } catch (err) {
        console.error('Create future state error:', err);
    }
}


async function createHypothesisFromKaizen() {
    if (!currentVSM || !currentVSM.project_id) {
        alert('Link this VSM to a project first (in VSM properties).');
        return;
    }
    if (!selectedElement || selectedElementType !== 'kaizen') return;

    const burstText = selectedElement.text || 'improvement';
    // Find nearest process step
    let nearestStep = 'the process';
    let minDist = Infinity;
    for (const step of (currentVSM.process_steps || [])) {
        const dx = (step.x || 0) - (selectedElement.x || 0);
        const dy = (step.y || 0) - (selectedElement.y || 0);
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < minDist) { minDist = dist; nearestStep = step.name || 'this step'; }
    }

    const description = `If we ${burstText.toLowerCase()} at ${nearestStep}, then lead time decreases and throughput improves`;
    try {
        const resp = await fetch(`/api/synara/${currentVSM.project_id}/hypotheses/add/`, {
            method: 'POST', credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                description: description,
                behavior_class: 'improvement',
                prior: 0.5
            })
        });
        if (!resp.ok) throw new Error('Failed to create hypothesis');
        const data = await resp.json();
        const h = data.hypothesis;
        selectedElement.hypothesis_id = h.id;
        await loadKaizenHypotheses(selectedElement);
        document.getElementById('prop-kaizen-hypothesis').value = h.id;
        renderVSM();
        saveVSM();
    } catch (e) {
        alert('Failed to create hypothesis: ' + safeStr(e, 'Unknown error'));
    }
}


async function createSelectedProposals() {
    const selected = proposalData.filter((_, i) => document.getElementById(`proposal-${i}`).checked);
    if (selected.length === 0) {
        alert('Select at least one proposal.');
        return;
    }

    try {
        const volume = document.getElementById('proposal-volume').value || 100000;
        const cost = document.getElementById('proposal-cost').value || 50;

        const response = await fetch('/api/hoshin/projects/from-proposals/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({
                vsm_id: vsmId,
                proposals: selected,
                annual_volume: parseFloat(volume),
                cost_per_unit: parseFloat(cost)
            })
        });

        if (!response.ok) {
            const data = await response.json();
            alert(safeStr(data.error, 'Failed to create projects.'));
            return;
        }

        const data = await response.json();
        closeProposalModal();
        alert(`Created ${data.created} CI project(s). View them in Hoshin Kanri.`);
    } catch (err) {
        alert('Error creating projects: ' + safeStr(err, 'Unknown error'));
    }
}


function exportVSM() {
    // TODO: PDF/image export
    alert('Export coming soon');
}

// =============================================================================
// CI Proposals (Enterprise)
// =============================================================================
let proposalData = [];


async function generateVSMProposals() {
    const volume = document.getElementById('proposal-volume').value || 100000;
    const cost = document.getElementById('proposal-cost').value || 50;

    document.getElementById('proposal-params').style.display = 'none';
    document.getElementById('proposal-loading').style.display = 'block';
    document.getElementById('proposal-error').style.display = 'none';

    try {
        const response = await fetch(`/api/vsm/${vsmId}/generate-proposals/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ annual_volume: parseFloat(volume), cost_per_unit: parseFloat(cost) })
        });

        const data = await response.json();
        if (!response.ok) {
            document.getElementById('proposal-loading').style.display = 'none';
            document.getElementById('proposal-error').style.display = 'block';
            document.getElementById('proposal-error').textContent = safeStr(data.error, 'Failed to generate proposals.');
            document.getElementById('proposal-params').style.display = 'block';
            return;
        }

        proposalData = data.proposals || [];
        document.getElementById('proposal-loading').style.display = 'none';

        if (proposalData.length === 0) {
            document.getElementById('proposal-error').style.display = 'block';
            document.getElementById('proposal-error').textContent = 'No proposals generated. Ensure kaizen bursts exist on the future state.';
            document.getElementById('proposal-params').style.display = 'block';
            return;
        }

        renderProposalCards(proposalData);
        document.getElementById('proposal-results').style.display = 'block';
    } catch (err) {
        document.getElementById('proposal-loading').style.display = 'none';
        document.getElementById('proposal-error').style.display = 'block';
        document.getElementById('proposal-error').textContent = 'Network error: ' + safeStr(err, 'Unknown error');
        document.getElementById('proposal-params').style.display = 'block';
    }
}


async function loadHypothesisProbabilities() {
    // Called after loading a VSM — enrich kaizen bursts with hypothesis probabilities
    if (!currentVSM || !currentVSM.project_id) return;
    const bursts = currentVSM.kaizen_bursts || [];
    const linked = bursts.filter(b => b.hypothesis_id);
    if (linked.length === 0) return;
    try {
        const resp = await fetch(`/api/synara/${currentVSM.project_id}/hypotheses/`, { credentials: 'same-origin' });
        if (!resp.ok) return;
        const data = await resp.json();
        const hMap = {};
        for (const h of (data.hypotheses || [])) { hMap[h.id] = h.posterior || h.prior || 0.5; }
        for (const b of linked) {
            b._hypothesis_prob = hMap[b.hypothesis_id] || null;
        }
    } catch (e) { /* optional enrichment */ }
}


// --- Hypothesis-driven kaizen tracking (Phase 5) ---

async function loadKaizenHypotheses(burst) {
    const select = document.getElementById('prop-kaizen-hypothesis');
    const probEl = document.getElementById('prop-kaizen-prob');
    select.innerHTML = '<option value="">— None —</option>';
    probEl.style.display = 'none';

    if (!currentVSM || !currentVSM.project_id) {
        select.innerHTML = '<option value="">— Link VSM to project first —</option>';
        return;
    }
    try {
        const resp = await fetch(`/api/synara/${currentVSM.project_id}/hypotheses/`, { credentials: 'same-origin' });
        if (!resp.ok) return;
        const data = await resp.json();
        const hypotheses = data.hypotheses || [];
        for (const h of hypotheses) {
            const opt = document.createElement('option');
            opt.value = h.id;
            opt.textContent = `${h.description.slice(0, 40)}${h.description.length > 40 ? '...' : ''} (P=${Math.round((h.posterior || h.prior || 0.5) * 100)}%)`;
            if (burst.hypothesis_id === h.id) opt.selected = true;
            select.appendChild(opt);
        }
        if (burst.hypothesis_id) {
            const linked = hypotheses.find(h => h.id === burst.hypothesis_id);
            if (linked) {
                const prob = linked.posterior || linked.prior || 0.5;
                probEl.textContent = `Current P(H) = ${Math.round(prob * 100)}%`;
                probEl.style.display = 'block';
            }
        }
    } catch (e) {
        // Silently fail — hypothesis linking is optional
    }
}


function openProposalModal() {
    const modal = document.getElementById('proposals-modal');
    modal.style.display = 'flex';
    // Reset to params step
    document.getElementById('proposal-params').style.display = 'block';
    document.getElementById('proposal-results').style.display = 'none';
    document.getElementById('proposal-loading').style.display = 'none';
    document.getElementById('proposal-error').style.display = 'none';
}


async function promoteVSM(vsmId) {
    if (!confirm('Promote this future-state VSM to current? The old current state will be archived.')) return;
    try {
        const resp = await fetch(`/api/hoshin/vsm/${vsmId}/promote/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCookie('csrftoken') },
            body: JSON.stringify({ action: 'promote' }),
        });
        if (!resp.ok) {
            const err = await resp.json();
            alert(safeStr(err.error, 'Promotion failed'));
            return;
        }
        alert('VSM promoted to current state. Old current archived.');
        location.reload();
    } catch (e) {
        console.error('Promote VSM error:', e);
    }
}

function renderProposalCards(proposals) {
    const container = document.getElementById('proposal-list');
    const priorityColors = { high: '#ef4444', medium: '#f59e0b', low: '#6b7280' };

    container.innerHTML = proposals.map((p, i) => `
        <div style="background:var(--bg-tertiary); border:1px solid var(--border); border-radius:4px; padding:1rem; margin-bottom:0.75rem;">
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
                <input type="checkbox" id="proposal-${i}" checked>
                <label for="proposal-${i}" style="font-weight:500; flex:1; cursor:pointer;">${p.suggested_title}</label>
                <span style="font-size:0.7rem; padding:0.15rem 0.5rem; border-radius:2px; background:${priorityColors[p.priority] || '#6b7280'}22; color:${priorityColors[p.priority] || '#6b7280'}; text-transform:uppercase;">${p.priority}</span>
            </div>
            <div style="font-size:0.8rem; color:var(--text-dim); margin-bottom:0.5rem;">
                Process: <strong style="color:var(--text-secondary);">${p.process_step}</strong>
                ${p.has_current_match ? '' : '<span style="color:var(--warning);"> (no baseline match)</span>'}
            </div>
            <div style="display:flex; gap:1rem; font-size:0.75rem; color:var(--text-dim);">
                ${p.metric_deltas.cycle_time ? `<span>C/T: ${p.metric_deltas.cycle_time > 0 ? '-' : '+'}${Math.abs(p.metric_deltas.cycle_time)}s</span>` : ''}
                ${p.metric_deltas.changeover ? `<span>C/O: ${p.metric_deltas.changeover > 0 ? '-' : '+'}${Math.abs(p.metric_deltas.changeover)}s</span>` : ''}
                ${p.metric_deltas.uptime ? `<span>Uptime: ${p.metric_deltas.uptime > 0 ? '+' : ''}${p.metric_deltas.uptime}%</span>` : ''}
            </div>
            <div style="margin-top:0.5rem; font-size:0.85rem;">
                Est. savings: <strong style="color:var(--success);">$${Math.round(p.lower_5 || 0).toLocaleString()} &mdash; $${Math.round(p.upper_95 || 0).toLocaleString()}</strong>/yr
                <span style="font-size:0.7rem; color:var(--text-dim); margin-left:0.25rem;">(90% CI, median $${Math.round(p.median_savings || p.estimated_annual_savings || 0).toLocaleString()})</span>
                ${p.p_positive > 0 ? `<span style="font-size:0.7rem; color:var(--success); margin-left:0.5rem;">${Math.round(p.p_positive * 100)}% chance of positive ROI</span>` : ''}
            </div>
            <button class="btn btn-primary" style="margin-top:0.5rem;font-size:0.75rem;padding:0.3rem 0.8rem;background:var(--accent);border-color:var(--accent);"
                onclick="approveProposal('${p.burst_id||''}', '${(p.suggested_title||'').replace(/'/g,'\\&#39;')}', ${Math.round(p.median_savings || p.estimated_annual_savings || 0)}, '${p.suggested_method||'direct'}', '${p.suggested_type||'material'}')">
                Approve → Hoshin Project
            </button>
        </div>
    `).join('');
}


function showTimeline() {
    if (!currentVSM) return;
    const snaps = currentVSM.metric_snapshots || [];
    const modal = document.getElementById('timeline-modal');
    modal.style.display = 'flex';

    if (snaps.length < 2) {
        document.getElementById('timeline-chart').style.display = 'none';
        document.getElementById('timeline-empty').style.display = 'block';
        return;
    }
    document.getElementById('timeline-chart').style.display = 'block';
    document.getElementById('timeline-empty').style.display = 'none';

    const timestamps = snaps.map(s => s.timestamp ? new Date(s.timestamp).toLocaleString() : '');
    const leadTimes = snaps.map(s => s.lead_time || 0);
    const pces = snaps.map(s => s.pce || 0);

    const traces = [
        { x: timestamps, y: leadTimes, name: 'Lead Time (days)', type: 'scatter', mode: 'lines+markers', line: { color: '#e74c3c' } },
        { x: timestamps, y: pces, name: 'PCE (%)', type: 'scatter', mode: 'lines+markers', yaxis: 'y2', line: { color: '#4a9f6e' } }
    ];
    const layout = {
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: '#999', size: 11 },
        margin: { t: 30, b: 60, l: 60, r: 60 },
        xaxis: { showgrid: false },
        yaxis: { title: 'Lead Time (days)', gridcolor: 'rgba(255,255,255,0.05)' },
        yaxis2: { title: 'PCE (%)', overlaying: 'y', side: 'right', gridcolor: 'rgba(255,255,255,0.05)' },
        legend: { orientation: 'h', y: -0.2 },
        showlegend: true
    };
    Plotly.newPlot('timeline-chart', traces, layout, { responsive: true, displayModeBar: false });
}


function updateProposalButton() {
    const btn = document.getElementById('btn-generate-proposals');
    if (!btn) return;
    const user = window.svendUser;
    if (user && user.features && user.features.hoshin_kanri && currentVSM && currentVSM.status !== 'future') {
        btn.style.display = 'block';
    } else {
        btn.style.display = 'none';
    }
}

// Also check when user data arrives (may happen before or after VSM load)
window.addEventListener('svendUserReady', function() { updateProposalButton(); });

