// DOE Workbench — Chat Modal
// Load order: doe-state.js → doe-design.js → doe-analysis.js → doe-optimize.js → doe-power.js → doe-chat.js

// =============================================================================
// DOE Guidance Chat
// =============================================================================

async function loadAvailableModels() {
    try {
        const response = await fetch('/api/experimenter/models/', { credentials: 'include' });
        const data = await response.json();
        selectedModel = data.default;
    } catch (err) {
        console.error('Failed to load models:', err);
    }
}

// Legacy compat — old code may call toggleChatModal
function toggleChatModal() {
    if (!contextPanelOpen) toggleContextPanel();
    showContextTab('chat');
}

function updateContextBadge() {
    // Update both panel badge and any legacy badge
    const badges = document.querySelectorAll('#context-badge, #panel-context-badge');
    badges.forEach(badge => {
        if (currentAnalysis) {
            badge.textContent = `Analysis: ${currentAnalysis.analysis?.model_summary?.r_squared || '?'}% R²`;
            badge.className = 'context-badge has-analysis';
        } else if (currentDesign) {
            badge.textContent = `Design: ${currentDesign.name || currentDesign.design_type} (${currentDesign.runs?.length || 0} runs)`;
            badge.className = 'context-badge has-design';
        } else {
            badge.textContent = 'No design';
            badge.className = 'context-badge';
        }
    });
}

function getSessionContext() {
    const context = {
        step: currentStep,
    };

    if (currentDesign) {
        context.design = {
            name: currentDesign.name,
            design_type: currentDesign.design_type,
            factors: currentDesign.factors,
            properties: currentDesign.properties,
        };
    }

    if (currentAnalysis) {
        context.analysis = {
            model_summary: currentAnalysis.analysis?.model_summary,
            significant_effects: currentAnalysis.analysis?.coefficients
                ?.filter(c => c.significant)
                ?.map(c => c.term) || [],
        };
    }

    const factors = [];
    document.querySelectorAll('.factor-row').forEach(row => {
        const name = row.querySelector('.factor-name')?.value?.trim();
        const levels = row.querySelector('.factor-levels')?.value?.trim();
        if (name) {
            factors.push({ name, levels });
        }
    });
    if (factors.length > 0) {
        context.factors = factors;
    }

    return context;
}

function handleChatKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChatMessage();
    }
}

function autoResizeInput(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

function askQuickQuestion(question) {
    const input = document.getElementById('panel-chat-input');
    if (input) input.value = question;
    // Ensure panel is open on chat tab
    if (!contextPanelOpen) toggleContextPanel();
    showContextTab('chat');
    sendChatMessage();
}

async function sendChatMessage() {
    const input = document.getElementById('panel-chat-input');
    const message = input.value.trim();

    if (!message) return;

    input.value = '';
    input.style.height = 'auto';

    const welcome = document.querySelector('.chat-welcome');
    if (welcome) welcome.remove();

    addChatMessage('user', message);

    chatHistory.push({ role: 'user', content: message });

    const typingId = showTypingIndicator();

    try {
        const response = await fetch('/api/experimenter/chat/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({
                message,
                context: getSessionContext(),
                history: chatHistory.slice(-20),
                model: selectedModel,
            }),
        });

        removeTypingIndicator(typingId);

        const data = await response.json();

        if (data.success) {
            addChatMessage('assistant', data.response);
            chatHistory.push({ role: 'assistant', content: data.response });
        } else {
            addChatMessage('assistant', `Sorry, I encountered an error: ${data.error}`);
        }
    } catch (err) {
        removeTypingIndicator(typingId);
        addChatMessage('assistant', `Sorry, I couldn't process your request. Please try again.`);
        console.error('Chat error:', err);
    }
}

function addChatMessage(role, content) {
    const container = document.getElementById('panel-chat-messages');

    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-message ${role}`;

    if (role === 'assistant') {
        content = content.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        content = content.replace(/\*(.+?)\*/g, '<em>$1</em>');
        content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
        content = content.replace(/^[•-]\s+(.+)$/gm, '<li>$1</li>');
        content = content.replace(/(<li>.*?<\/li>\n?)+/gs, '<ul>$&</ul>');
        content = content.replace(/^#{1,2}\s+(.+)$/gm, '<h4>$1</h4>');
        content = content.replace(/^#{3,}\s+(.+)$/gm, '<h5>$1</h5>');
        content = content.replace(/\n\n/g, '</p><p>');
        content = content.replace(/\n/g, '<br>');
        if (!content.startsWith('<')) {
            content = `<p>${content}</p>`;
        }
        msgDiv.innerHTML = content;
    } else {
        msgDiv.textContent = content;
    }

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function showTypingIndicator() {
    const container = document.getElementById('panel-chat-messages');
    const id = 'typing-' + Date.now();

    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message assistant typing';
    typingDiv.id = id;
    typingDiv.innerHTML = '<span></span><span></span><span></span>';

    container.appendChild(typingDiv);
    container.scrollTop = container.scrollHeight;

    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}
