/**
 * VSM List Module
 * VSM selection list view, new VSM dialog.
 * MIGRATION: Extracted from templates/vsm.html
 */


function hideNewVSMDialog() {
    document.getElementById('new-vsm-dialog').style.display = 'none';
}

// =============================================================================
function showNewVSMDialog() {
    document.getElementById('new-vsm-dialog').style.display = 'flex';
}


function showVSMList(maps) {
    const container = document.getElementById('canvas-container');
    const emptyState = document.getElementById('empty-state');

    // Group by fiscal year
    const groups = {};
    maps.forEach(m => {
        const fy = m.fiscal_year || 'Unscoped';
        if (!groups[fy]) groups[fy] = [];
        groups[fy].push(m);
    });

    // Sort FY keys: numbered years descending, then "Unscoped" last
    const fyKeys = Object.keys(groups).sort((a, b) => {
        if (a === 'Unscoped') return 1;
        if (b === 'Unscoped') return -1;
        return parseInt(b) - parseInt(a);
    });

    let html = '<div class="vsm-list"><h3 style="margin-bottom:1rem;">Your Value Stream Maps</h3>';

    fyKeys.forEach(fy => {
        const fyLabel = fy === 'Unscoped' ? 'Unscoped' : `FY ${fy}`;
        html += `<div style="margin-bottom:1.5rem;">
            <h4 style="font-size:13px;color:var(--text-secondary);margin:0 0 8px;text-transform:uppercase;letter-spacing:0.5px;">${fyLabel}</h4>`;

        groups[fy].forEach(m => {
            const promoteBtn = m.status === 'future' && m.paired_with_id
                ? `<button class="btn btn-outline" style="font-size:10px;padding:2px 8px;margin-left:8px;" onclick="event.stopPropagation();promoteVSM('${m.id}')">Promote to Current</button>`
                : '';
            const pairedLabel = m.paired_with_id
                ? '<span style="font-size:10px;color:var(--text-dim);margin-left:6px;">paired</span>'
                : '';

            html += `
                <div class="vsm-item" onclick="window.location.href='/app/vsm/${m.id}/'">
                    <div class="vsm-title">${m.name}
                        <span class="vsm-status ${m.status}">${m.status}</span>${pairedLabel}${promoteBtn}
                    </div>
                    <div class="vsm-meta">${m.product_family || 'No product family'} &middot; Updated ${new Date(m.updated_at).toLocaleDateString()}</div>
                </div>
            `;
        });
        html += '</div>';
    });
    html += '</div>';

    emptyState.innerHTML = html + '<button class="btn btn-primary" style="margin:1rem;" onclick="showNewVSMDialog()">+ New VSM</button>';
}

