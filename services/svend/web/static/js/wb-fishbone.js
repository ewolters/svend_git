// ============================================================================
// wb-fishbone.js — fishbone diagram logic
// ============================================================================

function createFishbone(x, y, effect) {
    const id = `fishbone-${++elementIdCounter}`;

    // Standard 6M categories
    const categories = [
        { name: 'People', causes: ['Training', 'Skills'] },
        { name: 'Process', causes: ['Procedure', 'Method'] },
        { name: 'Equipment', causes: ['Tools', 'Machines'] },
        { name: 'Materials', causes: ['Quality', 'Supply'] },
        { name: 'Environment', causes: ['Conditions', 'Layout'] },
        { name: 'Measurement', causes: ['Accuracy', 'Calibration'] }
    ];

    // Store fishbone data
    const fishboneData = {
        id,
        type: 'fishbone',
        x,
        y,
        effect,
        categories: categories.map(c => ({
            name: c.name,
            causes: [...c.causes]
        }))
    };
    elements.push(fishboneData);

    // Create DOM element
    const el = document.createElement('div');
    el.className = 'wb-element wb-fishbone';
    el.dataset.id = id;
    el.dataset.type = 'fishbone';
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;

    renderFishbone(el, fishboneData);

    // Make draggable
    el.addEventListener('mousedown', (e) => {
        if (e.target.contentEditable === 'true' || e.target.classList.contains('wb-fishbone-add')) return;
        onElementMouseDown(e, el);
    });

    canvas.appendChild(el);
    return el;
}

function renderFishbone(el, data) {
    const spineLength = 700;
    const boneSpacing = spineLength / 4;

    let html = `
        <div class="wb-fishbone-spine"></div>
        <div class="wb-fishbone-head" contenteditable="true"
             onblur="updateFishboneEffect('${data.id}', this.textContent)">${data.effect}</div>
    `;

    data.categories.forEach((cat, i) => {
        const isTop = i < 3;
        const xPos = 80 + (i % 3) * boneSpacing;
        const boneHeight = 120;

        // Bone line
        html += `
            <div class="wb-fishbone-bone ${isTop ? 'top' : 'bottom'}"
                 style="left: ${xPos}px; ${isTop ? 'bottom' : 'top'}: 50%; height: ${boneHeight}px;">
            </div>
        `;

        // Category label position
        const catY = isTop ? -boneHeight - 30 : boneHeight + 10;
        const catX = xPos + (isTop ? -50 : -50);

        html += `
            <div class="wb-fishbone-category"
                 style="left: ${xPos - 40}px; ${isTop ? 'top: 60px' : 'bottom: 60px'};"
                 contenteditable="true"
                 onblur="updateFishboneCategory('${data.id}', ${i}, this.textContent)">${cat.name}</div>
        `;

        // Causes container
        html += `
            <div class="wb-fishbone-causes ${isTop ? 'top' : 'bottom'}"
                 style="left: ${xPos - 30}px; ${isTop ? 'top: 100px' : 'bottom: 100px'};"
                 data-category="${i}">
        `;

        cat.causes.forEach((cause, j) => {
            html += `
                <div class="wb-fishbone-cause" contenteditable="true"
                     onblur="updateFishboneCause('${data.id}', ${i}, ${j}, this.textContent)">${cause}</div>
            `;
        });

        html += `
                <button class="wb-fishbone-add" onclick="addFishboneCause('${data.id}', ${i})">+ Add cause</button>
            </div>
        `;
    });

    el.innerHTML = html;
}

function updateFishboneEffect(id, text) {
    const data = elements.find(e => e.id === id);
    if (data) {
        saveState();
        data.effect = text;
    }
}

function updateFishboneCategory(id, catIndex, text) {
    const data = elements.find(e => e.id === id);
    if (data && data.categories[catIndex]) {
        saveState();
        data.categories[catIndex].name = text;
    }
}

function updateFishboneCause(id, catIndex, causeIndex, text) {
    const data = elements.find(e => e.id === id);
    if (data && data.categories[catIndex]) {
        saveState();
        if (text.trim() === '') {
            // Remove empty causes
            data.categories[catIndex].causes.splice(causeIndex, 1);
            const el = document.querySelector(`[data-id="${id}"]`);
            if (el) renderFishbone(el, data);
        } else {
            data.categories[catIndex].causes[causeIndex] = text;
        }
    }
}

function addFishboneCause(id, catIndex) {
    const data = elements.find(e => e.id === id);
    if (data && data.categories[catIndex]) {
        saveState();
        data.categories[catIndex].causes.push('New cause');
        const el = document.querySelector(`[data-id="${id}"]`);
        if (el) {
            renderFishbone(el, data);
            // Focus the new cause
            const causes = el.querySelectorAll(`[data-category="${catIndex}"] .wb-fishbone-cause`);
            const lastCause = causes[causes.length - 1];
            if (lastCause) {
                lastCause.focus();
                // Select all text
                const range = document.createRange();
                range.selectNodeContents(lastCause);
                const sel = window.getSelection();
                sel.removeAllRanges();
                sel.addRange(range);
            }
        }
    }
}
