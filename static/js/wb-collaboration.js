// ============================================================================
// wb-collaboration.js — polling, merge, save to server
// ============================================================================

let isCollaborative = false;
let roomCode = null;
let serverVersion = 0;
let pollInterval = null;
let votingActive = false;
let votesPerUser = 3;
let userVotes = [];
let myColor = '#4a9f6e';
let isOwner = false;
let lastCursorUpdate = 0;

// Guest mode (set from Django template context)
let isGuestMode = false;
let guestToken = window._wbGuestToken || '';
let guestPermission = window._wbGuestPermission || '';
let guestDisplayName = window._wbGuestDisplayName || '';
let guestRoomCode = window._wbGuestRoomCode || '';

if (guestToken) {
    isGuestMode = true;

    // Override fetch to inject guest token on whiteboard API calls
    const _guestOriginalFetch = window.fetch;
    window.fetch = function(url, options) {
        options = options || {};
        if (typeof url === 'string' && url.startsWith('/api/whiteboard/')) {
            if (!options.headers) options.headers = {};
            if (typeof options.headers === 'object' && !(options.headers instanceof Headers)) {
                options.headers['X-Guest-Token'] = guestToken;
            }
        }
        return _guestOriginalFetch.call(this, url, options);
    };
}

// Check if we're in a room (URL has room code)
function initCollaboration() {
    if (isGuestMode) {
        roomCode = guestRoomCode.toUpperCase();
        if (roomCode) loadBoardFromServer();
        return;
    }

    const pathParts = window.location.pathname.split('/');
    const roomIdx = pathParts.indexOf('whiteboard');
    if (roomIdx >= 0 && pathParts[roomIdx + 1] && pathParts[roomIdx + 1].length >= 4) {
        roomCode = pathParts[roomIdx + 1].toUpperCase();
        loadBoardFromServer();
    }
}

async function loadBoardFromServer() {
    try {
        const response = await fetch(`/api/whiteboard/boards/${roomCode}/`);
        if (!response.ok) {
            if (response.status === 404) {
                alert('Board not found. It may have been deleted.');
                window.location.href = '/app/whiteboard/';
                return;
            }
            throw new Error('Failed to load board');
        }

        const data = await response.json();
        isCollaborative = true;
        serverVersion = data.version;
        votingActive = data.voting_active;
        votesPerUser = data.votes_per_user;
        userVotes = data.user_votes || [];
        myColor = data.my_color;
        isOwner = data.is_owner;

        // Set project from loaded board (for hypothesis export compat)
        const boardProjectId = data.project_id || (data.project && data.project.id);
        if (boardProjectId) {
            currentProjectId = boardProjectId;
        }

        // Update UI
        document.getElementById('collab-controls').style.display = 'flex';
        document.getElementById('room-code-text').textContent = roomCode;
        if (isOwner) {
            document.getElementById('vote-toggle').style.display = 'block';
            document.getElementById('guest-manage-group').style.display = 'flex';
        }

        // Clear and load board
        clearCanvas(true);
        connections = data.connections || [];
        zoom = data.zoom || 1;
        panX = data.pan_x || 0;
        panY = data.pan_y || 0;

        updateCanvasTransform();
        document.getElementById('zoom-display').textContent = `${Math.round(zoom * 100)}%`;

        // Recreate elements
        (data.elements || []).forEach(el => {
            const domEl = createElement(el.type, el.x, el.y, el);
            elementIdCounter = Math.max(elementIdCounter, parseInt(el.id.split('-')[1]) || 0);

            // Add vote count if exists
            const voteCount = (data.vote_counts && data.vote_counts[el.id]) || 0;
            if (domEl) updateVoteDisplay(domEl, voteCount);
        });

        // Recreate connections
        setTimeout(() => renderConnections(), 100);

        // Update participants
        updateParticipantsList(data.participants);

        // Update voting UI
        updateVotingUI();

        // Start polling
        startPolling();

        showSyncStatus('synced', 'Connected');

        // Apply guest mode restrictions after board loads
        if (isGuestMode) applyGuestMode();

    } catch (err) {
        console.error('Error loading board:', err);
        showSyncStatus('error', 'Failed to connect');
    }
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(pollForUpdates, 2000);
}

async function pollForUpdates() {
    if (!isCollaborative || !roomCode) return;

    try {
        const response = await fetch(`/api/whiteboard/boards/${roomCode}/`);
        if (!response.ok) return;

        const data = await response.json();

        // Check if we need to update — skip if we have unsaved local changes
        if (data.version > serverVersion && !saveTimeout) {
            serverVersion = data.version;
            mergeServerState(data);
        }

        // Always update participants and votes
        updateParticipantsList(data.participants);
        if (data.vote_counts) updateAllVoteCounts(data.vote_counts);
        userVotes = data.user_votes || [];
        votingActive = data.voting_active;
        updateVotingUI();

        showSyncStatus('synced', 'Synced');
    } catch (err) {
        showSyncStatus('error', 'Connection lost');
    }
}

function mergeServerState(data) {
    clearCanvas(false);
    connections = data.connections || [];

    (data.elements || []).forEach(el => {
        const domEl = createElement(el.type, el.x, el.y, el);
        elementIdCounter = Math.max(elementIdCounter, parseInt(el.id.split('-')[1]) || 0);
    });

    setTimeout(() => renderConnections(), 50);
}

async function saveBoardToServer() {
    if (!isCollaborative || !roomCode) return;

    showSyncStatus('syncing', 'Saving...');

    try {
        const payload = {
            version: serverVersion,
            elements: elements,
            connections: connections,
            zoom: zoom,
            pan_x: panX,
            pan_y: panY,
        };
        if (currentProjectId) {
            payload.project_id = currentProjectId;
        }
        const response = await fetch(`/api/whiteboard/boards/${roomCode}/update/`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (response.status === 409) {
            const data = await response.json();
            serverVersion = data.server_version;
            mergeServerState(data);
            showSyncStatus('synced', 'Merged');
        } else if (response.ok) {
            const data = await response.json();
            serverVersion = data.version;
            showSyncStatus('synced', 'Saved');
        }
    } catch (err) {
        showSyncStatus('error', 'Save failed');
    }
}

// Debounced save
let saveTimeout = null;
function debouncedSave() {
    if (!isCollaborative) return;
    if (saveTimeout) clearTimeout(saveTimeout);
    saveTimeout = setTimeout(saveBoardToServer, 500);
}

// Override saveState to also sync
const originalSaveState = saveState;
saveState = function() {
    originalSaveState();
    debouncedSave();
};

async function shareBoard() {
    if (isCollaborative) {
        copyRoomLink();
        return;
    }

    const name = await svendPrompt('Board name:', 'Untitled Board');
    if (!name) return;

    try {
        const payload = { name };
        if (currentProjectId) {
            payload.project_id = currentProjectId;
        }
        const response = await fetch('/api/whiteboard/boards/create/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!response.ok) throw new Error('Failed to create board');

        const data = await response.json();
        roomCode = data.room_code;
        isCollaborative = true;
        isOwner = true;
        serverVersion = 0;

        await saveBoardToServer();

        window.history.pushState({}, '', `/app/whiteboard/${roomCode}/`);
        document.getElementById('collab-controls').style.display = 'flex';
        document.getElementById('room-code-text').textContent = roomCode;
        document.getElementById('vote-toggle').style.display = 'block';

        startPolling();
        copyRoomLink();
        alert(`Board created! Link copied to clipboard.\n${window.location.href}`);

    } catch (err) {
        console.error('Error creating board:', err);
        alert('Failed to create shared board.');
    }
}

function copyRoomLink() {
    const url = window.location.href;
    navigator.clipboard.writeText(url).then(() => {
        const display = document.getElementById('room-code-text');
        const original = display.textContent;
        display.textContent = 'Copied!';
        setTimeout(() => { display.textContent = original; }, 1500);
    });
}

function updateParticipantsList(participants) {
    const list = document.getElementById('participants-list');
    if (!participants || participants.length === 0) {
        list.innerHTML = '';
        return;
    }

    list.innerHTML = participants.map(p => `
        <div class="wb-participant" style="background:${p.color}" title="${p.username}${p.is_owner ? ' (owner)' : ''}">
            ${p.username.charAt(0).toUpperCase()}
        </div>
    `).join('');
}

function showSyncStatus(status, text) {
    let el = document.querySelector('.wb-sync-status');
    if (!el) {
        el = document.createElement('div');
        el.className = 'wb-sync-status';
        el.innerHTML = '<span class="wb-sync-dot"></span><span class="wb-sync-text"></span>';
        document.body.appendChild(el);
    }

    el.className = 'wb-sync-status ' + status;
    el.querySelector('.wb-sync-text').textContent = text;

    if (status === 'synced') {
        setTimeout(() => { el.style.opacity = '0.5'; }, 2000);
    } else {
        el.style.opacity = '1';
    }
}

// ============================================================================
// VOTING
// ============================================================================

async function toggleVoting() {
    if (!isCollaborative || !isOwner) return;

    const newState = !votingActive;
    const clearVotes = newState && confirm('Start new voting round? This will clear existing votes.');

    try {
        const response = await fetch(`/api/whiteboard/boards/${roomCode}/voting/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                active: newState,
                clear_votes: clearVotes,
            }),
        });

        if (response.ok) {
            const data = await response.json();
            votingActive = data.voting_active;
            votesPerUser = data.votes_per_user;
            updateVotingUI();
        }
    } catch (err) {
        console.error('Error toggling voting:', err);
    }
}

function updateVotingUI() {
    const btn = document.getElementById('vote-toggle');
    const remaining = document.getElementById('votes-remaining');

    if (votingActive) {
        btn.classList.add('active');
        remaining.style.display = 'block';
        remaining.textContent = `${votesPerUser - userVotes.length} votes left`;

        document.querySelectorAll('.wb-element').forEach(el => {
            el.classList.add('voting-mode');
            el.classList.toggle('voted', userVotes.includes(el.id));
        });
    } else {
        btn.classList.remove('active');
        remaining.style.display = 'none';
        document.querySelectorAll('.wb-element').forEach(el => {
            el.classList.remove('voting-mode', 'voted');
        });
    }
}

function updateVoteDisplay(element, count) {
    if (!element) return;
    let badge = element.querySelector('.wb-vote-count');
    if (!badge) {
        badge = document.createElement('div');
        badge.className = 'wb-vote-count';
        element.appendChild(badge);
    }
    badge.textContent = count;
    badge.classList.toggle('zero', count === 0);
}

function updateAllVoteCounts(voteCounts) {
    document.querySelectorAll('.wb-element').forEach(el => {
        const count = voteCounts[el.id] || 0;
        updateVoteDisplay(el, count);
    });
}

async function voteOnElement(elementId) {
    if (!votingActive || !isCollaborative) return;

    if (userVotes.includes(elementId)) {
        // Remove vote
        try {
            const response = await fetch(`/api/whiteboard/boards/${roomCode}/vote/${elementId}/`, {
                method: 'DELETE',
            });
            if (response.ok) {
                const data = await response.json();
                userVotes = userVotes.filter(id => id !== elementId);
                updateVoteDisplay(document.getElementById(elementId), data.vote_count);
                updateVotingUI();
            }
        } catch (err) {
            console.error('Error removing vote:', err);
        }
    } else {
        // Add vote
        if (userVotes.length >= votesPerUser) {
            alert('No votes remaining. Remove a vote first.');
            return;
        }
        try {
            const response = await fetch(`/api/whiteboard/boards/${roomCode}/vote/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ element_id: elementId }),
            });
            if (response.ok) {
                const data = await response.json();
                userVotes.push(elementId);
                updateVoteDisplay(document.getElementById(elementId), data.vote_count);
                updateVotingUI();
            } else {
                const err = await response.json();
                alert(safeStr(err.error, 'Failed to vote'));
            }
        } catch (err) {
            console.error('Error voting:', err);
        }
    }
}

// Voting click handler
canvasContainer.addEventListener('click', (e) => {
    if (!votingActive) return;
    const element = e.target.closest('.wb-element');
    if (element) {
        e.preventDefault();
        e.stopPropagation();
        voteOnElement(element.id);
    }
}, true);

// ============================================================================
// Guest Mode
// ============================================================================

function applyGuestMode() {
    // Hide sidebar sections not relevant to guests
    document.querySelectorAll('.wb-sidebar-section').forEach(sec => {
        const header = sec.querySelector('.wb-sidebar-header span');
        if (!header) return;
        const label = header.textContent.trim();
        if (label === 'Boards' || label === 'Convert') {
            sec.style.display = 'none';
        }
        if (guestPermission === 'view' && (label === 'Shapes' || label === 'Templates')) {
            sec.style.display = 'none';
        }
    });

    // Hide project selector
    const projGroup = document.querySelector('.wb-project-group');
    if (projGroup) projGroup.style.display = 'none';

    // Hide share button
    const shareGroup = document.getElementById('share-group');
    if (shareGroup) shareGroup.style.display = 'none';

    // Hide export buttons
    document.querySelectorAll('[onclick*="exportCausal"], [onclick*="exportBoard"], [onclick*="exportBoardJSON"]').forEach(
        btn => { btn.style.display = 'none'; }
    );

    // View-only: disable editing tools
    if (guestPermission === 'view') {
        document.querySelectorAll('.wb-tool-btn[data-tool]').forEach(btn => {
            const tool = btn.getAttribute('data-tool');
            if (tool !== 'select' && tool !== 'pan') {
                btn.style.opacity = '0.3';
                btn.style.pointerEvents = 'none';
            }
        });
        // No-op save for view-only
        saveBoardToServer = function() { return Promise.resolve(); };
        debouncedSave = function() {};
    }

    // Show name entry if no display name set yet
    if (!guestDisplayName) {
        showGuestNameModal();
    }
}

function showGuestNameModal() {
    document.getElementById('guest-name-overlay').style.display = 'flex';
    setTimeout(() => document.getElementById('guest-name-input').focus(), 100);
}

async function submitGuestName() {
    const input = document.getElementById('guest-name-input');
    const name = input.value.trim();
    if (!name) {
        input.style.borderColor = '#d06060';
        return;
    }

    try {
        const response = await fetch(`/api/whiteboard/boards/${roomCode}/guest-name/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Guest-Token': guestToken },
            body: JSON.stringify({ display_name: name }),
        });

        if (response.ok) {
            guestDisplayName = name;
            document.getElementById('guest-name-overlay').style.display = 'none';
        } else {
            const data = await response.json();
            alert(safeStr(data.error, 'Failed to set name'));
        }
    } catch (err) {
        alert('Connection error. Please try again.');
    }
}

// Enter key submits guest name
document.getElementById('guest-name-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') submitGuestName();
});

// Guest invite management (for board owners)
async function openGuestModal() {
    document.getElementById('guest-modal-overlay').style.display = 'flex';
    await refreshGuestList();
}

function closeGuestModal() {
    document.getElementById('guest-modal-overlay').style.display = 'none';
}

async function refreshGuestList() {
    try {
        const response = await fetch(`/api/whiteboard/boards/${roomCode}/guests/`);
        if (!response.ok) return;
        const data = await response.json();
        const list = document.getElementById('guest-invite-list');

        if (data.invites.length === 0) {
            list.innerHTML = '<p style="color:var(--wb-text-dim);text-align:center;padding:20px;">No guest invites yet. Create one below.</p>';
            return;
        }

        list.innerHTML = data.invites.map(inv => {
            const permLabel = inv.permission === 'edit_vote' ? 'Edit+Vote' : inv.permission === 'edit' ? 'Edit' : 'View';
            let statusBadge = '';
            if (!inv.is_active) statusBadge = '<span style="color:#d06060;font-size:10px;margin-left:6px;">revoked</span>';
            else if (inv.is_expired) statusBadge = '<span style="color:#e89547;font-size:10px;margin-left:6px;">expired</span>';
            else if (inv.is_online) statusBadge = '<span style="color:#4a9f6e;font-size:10px;margin-left:6px;">online</span>';

            const canRevoke = inv.is_active && !inv.is_expired;

            return `<div class="wb-guest-item">
                <div style="display:flex;align-items:center;gap:8px;min-width:0;">
                    <span style="width:10px;height:10px;border-radius:50%;background:${inv.color};display:inline-block;flex-shrink:0;"></span>
                    <strong style="font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${inv.display_name}</strong>
                    <span style="font-size:11px;color:var(--wb-text-dim);">${permLabel}</span>
                    ${statusBadge}
                </div>
                <div style="display:flex;gap:6px;flex-shrink:0;">
                    ${canRevoke ? `<button onclick="copyGuestLink('${inv.url}')" style="padding:3px 8px;font-size:11px;background:var(--wb-bg);color:var(--wb-text);border:1px solid var(--wb-border);border-radius:3px;cursor:pointer;">Copy</button>
                    <button onclick="revokeInvite('${inv.id}')" style="padding:3px 8px;font-size:11px;background:rgba(208,96,96,0.1);color:#d06060;border:1px solid rgba(208,96,96,0.3);border-radius:3px;cursor:pointer;">Revoke</button>` : ''}
                </div>
            </div>`;
        }).join('');
    } catch (err) {
        console.error('Failed to load guest list:', err);
    }
}

async function createGuestInvite() {
    const perm = document.getElementById('guest-perm-select').value;
    try {
        const response = await fetch(`/api/whiteboard/boards/${roomCode}/guests/create/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ permission: perm }),
        });

        if (response.ok) {
            const data = await response.json();
            const fullUrl = window.location.origin + data.url;
            navigator.clipboard.writeText(fullUrl).catch(() => {});
            const expDate = new Date(data.expires_at);
            const permLabel = perm === 'edit_vote' ? 'Edit+Vote' : perm === 'edit' ? 'Edit' : 'View';
            alert(`Guest invite created! Link copied.\n\nPermission: ${permLabel}\nExpires: ${expDate.toLocaleDateString()}\n\n${fullUrl}`);
            await refreshGuestList();
        } else {
            const data = await response.json();
            alert(safeStr(data.error, 'Failed to create invite'));
        }
    } catch (err) {
        alert('Connection error');
    }
}

async function revokeInvite(inviteId) {
    if (!confirm('Revoke this guest invite? They will lose access immediately.')) return;
    try {
        const response = await fetch(`/api/whiteboard/boards/${roomCode}/guests/${inviteId}/revoke/`, {
            method: 'DELETE',
        });
        if (response.ok) {
            await refreshGuestList();
        } else {
            alert('Failed to revoke invite');
        }
    } catch (err) {
        alert('Connection error');
    }
}

function copyGuestLink(url) {
    const fullUrl = window.location.origin + url;
    navigator.clipboard.writeText(fullUrl).then(() => {
        // Could show brief toast here
    });
}

// ============================================================================
// Save to Notebook — NB-001 §2.3
// ============================================================================

async function saveToNotebook() {
    if (elements.length === 0) {
        alert('Nothing to save. Add some elements first.');
        return;
    }
    if (!currentRoomCode) {
        alert('Save the board first (Ctrl+S) before adding to a notebook.');
        return;
    }

    // Fetch notebooks
    let notebooks = [];
    try {
        const res = await fetch('/api/notebooks/', { credentials: 'include', headers: { 'X-CSRFToken': getCSRFToken() } });
        if (res.ok) notebooks = (await res.json()).notebooks || [];
    } catch (e) { /* ignore */ }

    if (notebooks.length === 0) {
        alert('No notebooks found. Create a notebook first from the Notebooks page.');
        return;
    }

    // Build quick selector
    const opts = notebooks.map(nb => `<option value="${nb.id}">${nb.title} (${nb.status})</option>`).join('');
    const modal = document.createElement('div');
    modal.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.6);display:flex;align-items:center;justify-content:center;z-index:10000;';
    modal.innerHTML = `
        <div style="background:var(--bg-card,#1e1e1e);border:1px solid var(--border,#333);border-radius:8px;padding:24px;width:340px;">
            <h3 style="margin:0 0 16px;font-size:15px;">Save to Notebook</h3>
            <label style="font-size:12px;color:var(--text-dim,#aaa);display:block;margin-bottom:4px;">Notebook</label>
            <select id="nb-save-select" style="width:100%;padding:6px 8px;font-size:13px;background:var(--bg-primary,#111);color:var(--text,#eee);border:1px solid var(--border,#333);border-radius:4px;margin-bottom:16px;">
                ${opts}
            </select>
            <div style="display:flex;gap:8px;justify-content:flex-end;">
                <button onclick="this.closest('div[style*=fixed]').remove()" style="padding:6px 14px;font-size:12px;background:transparent;color:var(--text,#eee);border:1px solid var(--border,#333);border-radius:4px;cursor:pointer;">Cancel</button>
                <button id="nb-save-confirm" style="padding:6px 14px;font-size:12px;background:#5b9bd5;color:#fff;border:none;border-radius:4px;cursor:pointer;">Save</button>
            </div>
        </div>`;
    document.body.appendChild(modal);
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.remove(); });

    document.getElementById('nb-save-confirm').onclick = async () => {
        const notebookId = document.getElementById('nb-save-select').value;
        const btn = document.getElementById('nb-save-confirm');
        btn.disabled = true;
        btn.textContent = 'Saving...';

        try {
            // Get SVG from server (uses existing _generate_svg)
            const svgRes = await fetch(`/api/whiteboard/boards/${currentRoomCode}/svg/?theme=dark`, {
                credentials: 'include', headers: { 'X-CSRFToken': getCSRFToken() }
            });
            let svgHtml = '';
            if (svgRes.ok) {
                const svgData = await svgRes.json();
                svgHtml = svgData.svg || '';
            }

            // Collect element summary
            const elTypes = {};
            elements.forEach(el => { elTypes[el.type] = (elTypes[el.type] || 0) + 1; });

            const boardTitle = document.getElementById('board-title')?.value || 'Whiteboard';
            const payload = {
                page_type: 'note',
                title: `Whiteboard — ${boardTitle}`,
                source_tool: 'whiteboard',
                inputs: {
                    room_code: currentRoomCode,
                    elements_count: elements.length,
                    connections_count: connections.length,
                    element_types: elTypes,
                },
                outputs: {
                    board_name: boardTitle,
                    causal_links: connections.filter(c => c.type === 'causal').length,
                },
                rendered_html: svgHtml,
                narrative: `Whiteboard snapshot: ${elements.length} elements, ${connections.length} connections` +
                    (connections.filter(c => c.type === 'causal').length > 0
                        ? `, ${connections.filter(c => c.type === 'causal').length} causal (If-Then) links` : '') + '.',
            };

            const res = await fetch(`/api/notebooks/${notebookId}/pages/`, {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                body: JSON.stringify(payload),
            });

            if (res.ok) {
                btn.textContent = 'Saved!';
                btn.style.background = '#4a9f6e';
                setTimeout(() => modal.remove(), 800);
            } else {
                const err = await res.json().catch(() => ({}));
                alert('Failed to save: ' + (err.error || res.statusText));
                btn.disabled = false;
                btn.textContent = 'Save';
            }
        } catch (e) {
            alert('Failed to save: ' + e.message);
            btn.disabled = false;
            btn.textContent = 'Save';
        }
    };
}

// ============================================================================
// Initialize
// ============================================================================

if (!isGuestMode) {
    loadBoardsList();
}
saveState(); // Save initial empty state
initCollaboration(); // Check for room code in URL

// Feature gate check
if (!window._wbGuestToken) {
    window.addEventListener('svendUserReady', function(e) {
        if (!e.detail.features || !e.detail.features.full_tools) showUpgradeModal();
    });
}
