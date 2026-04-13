/**
 * calc-scheduling.js — Scheduling Tool Calculators
 *
 * Load order: after calc-core.js (uses SvendOps, renderNextSteps)
 * Extracted from: calculators.html (inline script)
 *
 * Contains: Job Sequencer, Sequence Optimizer, Capacity Load Chart,
 *           Mixed-Model Sequencer, Due Date Risk Simulator
 */

// ============================================================================
// SCHEDULING TOOLS
// ============================================================================

// ============================================================================
// Job Sequencer
// ============================================================================

let sequencerJobs = [
    { id: 1, name: 'Job A', processTime: 45, dueDate: 120, setupGroup: 'Type1' },
    { id: 2, name: 'Job B', processTime: 30, dueDate: 90, setupGroup: 'Type2' },
    { id: 3, name: 'Job C', processTime: 60, dueDate: 200, setupGroup: 'Type1' },
    { id: 4, name: 'Job D', processTime: 25, dueDate: 150, setupGroup: 'Type3' },
];

let sequencerOrder = [0, 1, 2, 3]; // Indices into sequencerJobs
let draggedJobIdx = null;

function renderSequencerJobs() {
    const container = document.getElementById('sequencer-jobs');
    if (!container) return;

    container.innerHTML = sequencerJobs.map((job, i) => `
        <div class="yamazumi-station" style="gap: 8px;">
            <input type="text" value="${job.name}" style="flex: 1; padding: 8px;"
                   oninput="updateSequencerJob(${i}, 'name', this.value)" placeholder="Job name">
            <input type="number" value="${job.processTime}" style="width: 60px; text-align: right;"
                   oninput="updateSequencerJob(${i}, 'processTime', this.value)" title="Process time">
            <span style="color: var(--text-dim); font-size: 11px;">min</span>
            <input type="number" value="${job.dueDate}" style="width: 70px; text-align: right;"
                   oninput="updateSequencerJob(${i}, 'dueDate', this.value)" title="Due date">
            <span style="color: var(--text-dim); font-size: 11px;">due</span>
            <input type="text" value="${job.setupGroup}" style="width: 60px; text-align: center;"
                   oninput="updateSequencerJob(${i}, 'setupGroup', this.value)" title="Setup group">
            <button class="yamazumi-station-remove" onclick="removeSequencerJob(${i})">&times;</button>
        </div>
    `).join('');

    renderSequencerGantt();
}

function addSequencerJob() {
    const id = sequencerJobs.length > 0 ? Math.max(...sequencerJobs.map(j => j.id)) + 1 : 1;
    sequencerJobs.push({ id, name: `Job ${String.fromCharCode(65 + sequencerJobs.length)}`, processTime: 30, dueDate: 180, setupGroup: 'Type1' });
    sequencerOrder.push(sequencerJobs.length - 1);
    renderSequencerJobs();
}

function removeSequencerJob(idx) {
    sequencerJobs.splice(idx, 1);
    // Update order indices
    sequencerOrder = sequencerOrder.filter(i => i !== idx).map(i => i > idx ? i - 1 : i);
    renderSequencerJobs();
}

function updateSequencerJob(idx, field, value) {
    if (field === 'processTime' || field === 'dueDate') value = parseFloat(value) || 0;
    sequencerJobs[idx][field] = value;
    renderSequencerGantt();
}

function loadSampleJobs() {
    sequencerJobs = [
        { id: 1, name: 'Order 101', processTime: 45, dueDate: 120, setupGroup: 'A' },
        { id: 2, name: 'Order 102', processTime: 30, dueDate: 90, setupGroup: 'B' },
        { id: 3, name: 'Order 103', processTime: 60, dueDate: 200, setupGroup: 'A' },
        { id: 4, name: 'Order 104', processTime: 25, dueDate: 150, setupGroup: 'C' },
        { id: 5, name: 'Order 105', processTime: 50, dueDate: 250, setupGroup: 'B' },
        { id: 6, name: 'Order 106', processTime: 35, dueDate: 180, setupGroup: 'A' },
    ];
    sequencerOrder = [0, 1, 2, 3, 4, 5];
    renderSequencerJobs();
}

function getSetupTime(fromGroup, toGroup) {
    if (fromGroup === toGroup) return 0;
    // Try to get from changeover matrix if available
    if (typeof changeoverMatrix !== 'undefined' && changeoverMatrix[fromGroup]?.[toGroup]) {
        return changeoverMatrix[fromGroup][toGroup];
    }
    // Default: 10 min for different groups
    return 10;
}

function calcSequencerMetrics() {
    if (sequencerJobs.length === 0) return { makespan: 0, flowTime: 0, setup: 0, tardiness: 0, lateCount: 0 };

    let currentTime = 0;
    let totalFlowTime = 0;
    let totalSetup = 0;
    let totalTardiness = 0;
    let lateCount = 0;
    let prevGroup = null;

    const schedule = [];

    sequencerOrder.forEach(idx => {
        const job = sequencerJobs[idx];
        if (!job) return;

        // Setup time
        const setup = prevGroup ? getSetupTime(prevGroup, job.setupGroup) : 0;
        totalSetup += setup;

        const startTime = currentTime + setup;
        const endTime = startTime + job.processTime;

        schedule.push({ job, startTime, endTime, setup });

        totalFlowTime += endTime;

        if (endTime > job.dueDate) {
            totalTardiness += (endTime - job.dueDate);
            lateCount++;
        }

        currentTime = endTime;
        prevGroup = job.setupGroup;
    });

    return {
        makespan: currentTime,
        flowTime: totalFlowTime,
        avgFlowTime: sequencerJobs.length > 0 ? totalFlowTime / sequencerJobs.length : 0,
        setup: totalSetup,
        tardiness: totalTardiness,
        lateCount,
        schedule
    };
}

function renderSequencerGantt() {
    const container = document.getElementById('sequencer-gantt');
    if (!container) return;

    const metrics = calcSequencerMetrics();

    // Update metrics display
    document.getElementById('seq-makespan').innerHTML = `${metrics.makespan}<span class="calc-result-unit">min</span>`;
    document.getElementById('seq-flowtime').innerHTML = `${metrics.flowTime}<span class="calc-result-unit">min</span>`;
    document.getElementById('seq-setup').innerHTML = `${metrics.setup}<span class="calc-result-unit">min</span>`;
    document.getElementById('seq-late').textContent = metrics.lateCount;
    document.getElementById('seq-tardiness').textContent = `${metrics.tardiness} min (${metrics.lateCount} jobs late)`;
    document.getElementById('seq-avg-flow').textContent = `${metrics.avgFlowTime.toFixed(0)} min`;

    if (sequencerJobs.length === 0) {
        container.innerHTML = '<div style="color: var(--text-dim); text-align: center; padding: 40px;">Add jobs to see Gantt chart</div>';
        return;
    }

    // Generate colors for setup groups
    const groups = [...new Set(sequencerJobs.map(j => j.setupGroup))];
    const groupColors = {};
    groups.forEach((g, i) => {
        const hue = (i * 137) % 360;
        groupColors[g] = `hsl(${hue}, 60%, 45%)`;
    });

    // Render Gantt
    const scale = Math.max(metrics.makespan, Math.max(...sequencerJobs.map(j => j.dueDate)));
    const barHeight = 32;

    let html = `<div style="position: relative; min-width: ${Math.max(600, scale * 2)}px;">`;

    // Time axis
    html += `<div style="height: 24px; border-bottom: 1px solid var(--border); margin-bottom: 8px; position: relative;">`;
    for (let t = 0; t <= scale; t += Math.ceil(scale / 10)) {
        const left = (t / scale) * 100;
        html += `<span style="position: absolute; left: ${left}%; font-size: 10px; color: var(--text-dim);">${t}</span>`;
    }
    html += `</div>`;

    // Bars
    metrics.schedule.forEach((item, i) => {
        const leftPct = (item.startTime / scale) * 100;
        const widthPct = (item.job.processTime / scale) * 100;
        const setupWidthPct = (item.setup / scale) * 100;
        const duePct = (item.job.dueDate / scale) * 100;
        const isLate = item.endTime > item.job.dueDate;

        html += `
            <div style="height: ${barHeight}px; margin-bottom: 4px; position: relative; display: flex; align-items: center;"
                 draggable="true" data-idx="${i}"
                 ondragstart="handleJobDragStart(event, ${i})"
                 ondragover="handleJobDragOver(event)"
                 ondrop="handleJobDrop(event, ${i})"
                 ondragend="handleJobDragEnd()">
                <!-- Setup bar -->
                ${item.setup > 0 ? `<div style="position: absolute; left: ${leftPct - setupWidthPct}%; width: ${setupWidthPct}%; height: 60%; background: repeating-linear-gradient(45deg, #666, #666 2px, transparent 2px, transparent 6px); border-radius: 2px;" title="Setup: ${item.setup} min"></div>` : ''}
                <!-- Job bar -->
                <div style="position: absolute; left: ${leftPct}%; width: ${widthPct}%; height: 80%; background: ${groupColors[item.job.setupGroup]}; border-radius: 4px; display: flex; align-items: center; padding: 0 8px; color: white; font-size: 11px; cursor: grab; box-shadow: ${isLate ? '0 0 0 2px #e74c3c' : 'none'};" title="${item.job.name}: ${item.startTime}-${item.endTime} min${isLate ? ' (LATE)' : ''}">
                    ${item.job.name}
                </div>
                <!-- Due date marker -->
                <div style="position: absolute; left: ${duePct}%; width: 2px; height: 100%; background: ${isLate ? '#e74c3c' : '#4a9f6e'};" title="Due: ${item.job.dueDate}"></div>
            </div>
        `;
    });

    html += `</div>`;

    // Legend
    html += `<div style="display: flex; gap: 16px; margin-top: 16px; font-size: 11px; flex-wrap: wrap;">`;
    groups.forEach(g => {
        html += `<span style="display: flex; align-items: center; gap: 4px;"><span style="width: 12px; height: 12px; background: ${groupColors[g]}; border-radius: 2px;"></span>${g}</span>`;
    });
    html += `<span style="display: flex; align-items: center; gap: 4px;"><span style="width: 12px; height: 12px; background: repeating-linear-gradient(45deg, #666, #666 2px, transparent 2px, transparent 4px);"></span>Setup</span>`;
    html += `</div>`;

    container.innerHTML = html;
}

function handleJobDragStart(event, idx) {
    draggedJobIdx = idx;
    event.dataTransfer.effectAllowed = 'move';
}

function handleJobDragOver(event) {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
}

function handleJobDrop(event, targetIdx) {
    event.preventDefault();
    if (draggedJobIdx === null || draggedJobIdx === targetIdx) return;

    // Reorder
    const item = sequencerOrder.splice(draggedJobIdx, 1)[0];
    sequencerOrder.splice(targetIdx, 0, item);
    renderSequencerGantt();
}

function handleJobDragEnd() {
    draggedJobIdx = null;
}

function pullJobsFromLineSim() {
    if (typeof lineOrders !== 'undefined' && lineOrders.length > 0) {
        sequencerJobs = lineOrders.map((o, i) => ({
            id: i + 1,
            name: `Order ${o.id || i + 1}`,
            processTime: o.quantity * 5, // Rough estimate
            dueDate: o.dueTime || 300,
            setupGroup: o.product || 'A'
        }));
        sequencerOrder = sequencerJobs.map((_, i) => i);
        renderSequencerJobs();
        showToast('Pulled orders from Line Simulator');
    } else {
        showToast('No orders in Line Simulator');
    }
}

function pushSequenceToLineSim() {
    // Convert sequencer jobs to line sim orders
    const products = [...new Set(sequencerJobs.map(j => j.setupGroup))];
    // Ensure products exist
    if (typeof lineProducts !== 'undefined') {
        products.forEach((p, i) => {
            if (!lineProducts.find(lp => lp.id === p)) {
                lineProducts.push({ id: p, name: `Product ${p}`, color: `hsl(${i * 137 % 360}, 60%, 50%)`, ctMultiplier: 1.0 });
            }
        });
    }

    // Create orders
    if (typeof lineOrders !== 'undefined') {
        lineOrders = sequencerOrder.map((idx, i) => {
            const job = sequencerJobs[idx];
            return {
                id: i + 1,
                product: job.setupGroup,
                quantity: Math.ceil(job.processTime / 5), // Convert back
                dueTime: job.dueDate
            };
        });
    }

    showCalc('line-sim');
    setTimeout(() => {
        if (typeof renderProducts === 'function') renderProducts();
        if (typeof renderOrders === 'function') renderOrders();
        if (typeof initLineSim === 'function') initLineSim();
    }, 100);
    showToast('Pushed sequence to Line Simulator');
}

// ============================================================================
// Sequence Optimizer
// ============================================================================

let optimizerJobs = [];
let optimizedOrder = [];

function pullFromSequencer() {
    optimizerJobs = JSON.parse(JSON.stringify(sequencerJobs));
    optimizedOrder = [...sequencerOrder];
    runSequenceOptimizer();
    showToast('Pulled from Job Sequencer');
}

function calcOrderMetrics(jobs, order) {
    if (jobs.length === 0) return { makespan: 0, flowTime: 0, setup: 0, tardiness: 0 };

    let currentTime = 0;
    let totalFlowTime = 0;
    let totalSetup = 0;
    let totalTardiness = 0;
    let prevGroup = null;

    order.forEach(idx => {
        const job = jobs[idx];
        if (!job) return;

        const setup = prevGroup ? getSetupTime(prevGroup, job.setupGroup) : 0;
        totalSetup += setup;

        currentTime += setup + job.processTime;
        totalFlowTime += currentTime;

        if (currentTime > job.dueDate) {
            totalTardiness += (currentTime - job.dueDate);
        }

        prevGroup = job.setupGroup;
    });

    return {
        makespan: currentTime,
        flowTime: totalFlowTime,
        avgFlowTime: jobs.length > 0 ? totalFlowTime / jobs.length : 0,
        setup: totalSetup,
        tardiness: totalTardiness
    };
}

function runSequenceOptimizer() {
    if (optimizerJobs.length === 0) {
        optimizerJobs = JSON.parse(JSON.stringify(sequencerJobs));
        optimizedOrder = [...sequencerOrder];
    }

    const objective = document.getElementById('opt-objective')?.value || 'setup';
    const algorithm = document.getElementById('opt-algorithm')?.value || 'nearest';

    // Calculate current metrics
    const currentMetrics = calcOrderMetrics(optimizerJobs, optimizedOrder);

    // Run optimization
    let newOrder;
    switch (algorithm) {
        case 'edd':
            newOrder = [...optimizedOrder].sort((a, b) => optimizerJobs[a].dueDate - optimizerJobs[b].dueDate);
            break;
        case 'spt':
            newOrder = [...optimizedOrder].sort((a, b) => optimizerJobs[a].processTime - optimizerJobs[b].processTime);
            break;
        case 'nearest':
            newOrder = nearestNeighborSequence(optimizerJobs, objective);
            break;
        case '2opt':
            newOrder = twoOptImprove(optimizerJobs, nearestNeighborSequence(optimizerJobs, objective), objective);
            break;
        default:
            newOrder = [...optimizedOrder];
    }

    const newMetrics = calcOrderMetrics(optimizerJobs, newOrder);

    // Update display
    document.getElementById('opt-current-seq').textContent = optimizedOrder.map(i => optimizerJobs[i]?.name || '?').join(' → ');
    document.getElementById('opt-current-setup').textContent = currentMetrics.setup + ' min';
    document.getElementById('opt-current-tard').textContent = currentMetrics.tardiness + ' min';
    document.getElementById('opt-current-make').textContent = currentMetrics.makespan + ' min';
    document.getElementById('opt-current-flow').textContent = currentMetrics.avgFlowTime.toFixed(0) + ' min';

    document.getElementById('opt-new-seq').textContent = newOrder.map(i => optimizerJobs[i]?.name || '?').join(' → ');
    document.getElementById('opt-new-setup').textContent = newMetrics.setup + ' min';
    document.getElementById('opt-new-tard').textContent = newMetrics.tardiness + ' min';
    document.getElementById('opt-new-make').textContent = newMetrics.makespan + ' min';
    document.getElementById('opt-new-flow').textContent = newMetrics.avgFlowTime.toFixed(0) + ' min';

    // Improvement summary with $ valuation
    let improvement = '';
    const seqHourlyCost = parseFloat(document.getElementById('seq-hourly-cost')?.value) || 0;
    const dollarSuffix = (minSaved) => seqHourlyCost > 0 ? ` = <strong>$${((minSaved / 60) * seqHourlyCost).toFixed(0)} saved</strong>` : '';
    switch (objective) {
        case 'setup':
            const setupSaved = currentMetrics.setup - newMetrics.setup;
            improvement = setupSaved > 0 ? `<span style="color: #4a9f6e;">${setupSaved} min setup saved (${((setupSaved/currentMetrics.setup)*100).toFixed(0)}% reduction)${dollarSuffix(setupSaved)}</span>` : 'Already optimal for setup';
            break;
        case 'tardiness':
            const tardSaved = currentMetrics.tardiness - newMetrics.tardiness;
            improvement = tardSaved > 0 ? `<span style="color: #4a9f6e;">${tardSaved} min tardiness reduced${dollarSuffix(tardSaved)}</span>` : 'Already optimal for tardiness';
            break;
        case 'makespan':
            const makeSaved = currentMetrics.makespan - newMetrics.makespan;
            improvement = makeSaved > 0 ? `<span style="color: #4a9f6e;">${makeSaved} min makespan reduced${dollarSuffix(makeSaved)}</span>` : 'Already optimal for makespan';
            break;
        case 'flowtime':
            const flowSaved = currentMetrics.avgFlowTime - newMetrics.avgFlowTime;
            improvement = flowSaved > 0 ? `<span style="color: #4a9f6e;">${flowSaved.toFixed(0)} min avg flow time reduced${dollarSuffix(flowSaved)}</span>` : 'Already optimal for flow time';
            break;
    }
    document.getElementById('opt-improvement').innerHTML = improvement;

    optimizedOrder = newOrder;
}

function pullChangeoverMatrix() {
    if (!changeoverMatrix || Object.keys(changeoverMatrix).length === 0) {
        showToast('No changeover data — configure Changeover Matrix first', 'warning');
        return;
    }
    const products = Object.keys(changeoverMatrix);
    showToast(`Using per-pair setup times for ${products.length} products from Changeover Matrix`);
    runSequenceOptimizer();
}

function nearestNeighborSequence(jobs, objective) {
    if (jobs.length === 0) return [];

    const remaining = jobs.map((_, i) => i);
    const result = [];
    let prevGroup = null;

    while (remaining.length > 0) {
        let bestIdx = 0;
        let bestScore = Infinity;

        remaining.forEach((jobIdx, i) => {
            const job = jobs[jobIdx];
            let score;
            switch (objective) {
                case 'setup':
                    score = prevGroup ? getSetupTime(prevGroup, job.setupGroup) : 0;
                    break;
                case 'tardiness':
                case 'edd':
                    score = job.dueDate;
                    break;
                case 'spt':
                case 'flowtime':
                    score = job.processTime;
                    break;
                default:
                    score = prevGroup ? getSetupTime(prevGroup, job.setupGroup) : 0;
            }
            if (score < bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        });

        const chosen = remaining.splice(bestIdx, 1)[0];
        result.push(chosen);
        prevGroup = jobs[chosen].setupGroup;
    }

    return result;
}

function twoOptImprove(jobs, order, objective) {
    let improved = true;
    let best = [...order];
    let bestScore = getObjectiveScore(jobs, best, objective);

    while (improved) {
        improved = false;
        for (let i = 0; i < best.length - 1; i++) {
            for (let j = i + 1; j < best.length; j++) {
                const newOrder = [...best];
                // Reverse segment between i and j
                const segment = newOrder.splice(i, j - i + 1).reverse();
                newOrder.splice(i, 0, ...segment);

                const score = getObjectiveScore(jobs, newOrder, objective);
                if (score < bestScore) {
                    best = newOrder;
                    bestScore = score;
                    improved = true;
                }
            }
        }
    }
    return best;
}

function getObjectiveScore(jobs, order, objective) {
    const m = calcOrderMetrics(jobs, order);
    switch (objective) {
        case 'setup': return m.setup;
        case 'tardiness': return m.tardiness;
        case 'makespan': return m.makespan;
        case 'flowtime': return m.avgFlowTime;
        default: return m.setup;
    }
}

function applyOptimizedSequence() {
    sequencerJobs = JSON.parse(JSON.stringify(optimizerJobs));
    sequencerOrder = [...optimizedOrder];
    showCalc('sequencer');
    setTimeout(() => renderSequencerJobs(), 100);
    showToast('Applied optimized sequence');
}

// ============================================================================
// Capacity Load Chart
// ============================================================================

let capacityOrders = [
    { id: 1, name: 'WO-001', hours: 6, startDay: 1 },
    { id: 2, name: 'WO-002', hours: 8, startDay: 1 },
    { id: 3, name: 'WO-003', hours: 4, startDay: 2 },
    { id: 4, name: 'WO-004', hours: 10, startDay: 3 },
    { id: 5, name: 'WO-005', hours: 5, startDay: 4 },
];

function renderCapacityOrders() {
    const container = document.getElementById('capacity-orders');
    if (!container) return;

    container.innerHTML = capacityOrders.map((order, i) => `
        <div class="yamazumi-station" style="gap: 8px;">
            <input type="text" value="${order.name}" style="flex: 1; padding: 8px;"
                   oninput="updateCapacityOrder(${i}, 'name', this.value)" placeholder="Work Order">
            <input type="number" value="${order.hours}" style="width: 60px; text-align: right;"
                   oninput="updateCapacityOrder(${i}, 'hours', this.value)" title="Hours required">
            <span style="color: var(--text-dim); font-size: 11px;">hrs</span>
            <input type="number" value="${order.startDay}" style="width: 50px; text-align: right;"
                   oninput="updateCapacityOrder(${i}, 'startDay', this.value)" title="Start day" min="1">
            <span style="color: var(--text-dim); font-size: 11px;">day</span>
            <button class="yamazumi-station-remove" onclick="removeCapacityOrder(${i})">&times;</button>
        </div>
    `).join('');

    calcCapacityLoad();
}

function addCapacityOrder() {
    const id = capacityOrders.length > 0 ? Math.max(...capacityOrders.map(o => o.id)) + 1 : 1;
    capacityOrders.push({ id, name: `WO-${String(id).padStart(3, '0')}`, hours: 4, startDay: 1 });
    renderCapacityOrders();
}

function removeCapacityOrder(idx) {
    capacityOrders.splice(idx, 1);
    renderCapacityOrders();
}

function updateCapacityOrder(idx, field, value) {
    if (field === 'hours' || field === 'startDay') value = parseFloat(value) || 1;
    capacityOrders[idx][field] = value;
    calcCapacityLoad();
}

function loadSampleWorkOrders() {
    capacityOrders = [
        { id: 1, name: 'WO-101', hours: 6, startDay: 1 },
        { id: 2, name: 'WO-102', hours: 8, startDay: 1 },
        { id: 3, name: 'WO-103', hours: 7, startDay: 2 },
        { id: 4, name: 'WO-104', hours: 10, startDay: 2 },
        { id: 5, name: 'WO-105', hours: 5, startDay: 3 },
        { id: 6, name: 'WO-106', hours: 6, startDay: 4 },
        { id: 7, name: 'WO-107', hours: 9, startDay: 5 },
        { id: 8, name: 'WO-108', hours: 4, startDay: 6 },
    ];
    renderCapacityOrders();
}

function pullFromSequencerToCapacity() {
    capacityOrders = sequencerJobs.map((job, i) => ({
        id: i + 1,
        name: job.name,
        hours: job.processTime / 60, // Convert minutes to hours
        startDay: Math.ceil((job.dueDate / 60) / 8) // Rough estimate
    }));
    renderCapacityOrders();
    showToast('Pulled from Job Sequencer');
}

function calcCapacityLoad() {
    const hoursPerDay = parseFloat(document.getElementById('cap-hours-day')?.value) || 8;
    const daysToShow = parseInt(document.getElementById('cap-days')?.value) || 10;
    const efficiency = (parseFloat(document.getElementById('cap-efficiency')?.value) || 85) / 100;

    const effectiveHours = hoursPerDay * efficiency;

    // Calculate load per day
    const loadByDay = {};
    for (let d = 1; d <= daysToShow; d++) {
        loadByDay[d] = { orders: [], total: 0 };
    }

    capacityOrders.forEach(order => {
        let remaining = order.hours;
        let day = order.startDay;

        while (remaining > 0 && day <= daysToShow) {
            if (!loadByDay[day]) loadByDay[day] = { orders: [], total: 0 };

            const available = effectiveHours - loadByDay[day].total;
            const allocated = Math.min(remaining, available > 0 ? available : remaining);

            loadByDay[day].orders.push({ name: order.name, hours: allocated });
            loadByDay[day].total += allocated;
            remaining -= allocated;
            day++;
        }
    });

    // Calculate totals
    const totalLoad = capacityOrders.reduce((sum, o) => sum + o.hours, 0);
    const totalCapacity = daysToShow * effectiveHours;
    const utilization = totalCapacity > 0 ? (totalLoad / totalCapacity) * 100 : 0;
    const overloadDays = Object.values(loadByDay).filter(d => d.total > effectiveHours).length;

    document.getElementById('cap-total-load').innerHTML = `${totalLoad.toFixed(1)}<span class="calc-result-unit">hrs</span>`;
    document.getElementById('cap-available').innerHTML = `${totalCapacity.toFixed(1)}<span class="calc-result-unit">hrs</span>`;
    document.getElementById('cap-utilization').innerHTML = `${utilization.toFixed(0)}<span class="calc-result-unit">%</span>`;
    document.getElementById('cap-overload-days').textContent = overloadDays;

    // Update derivation
    document.getElementById('capacity-load-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Effective Daily Capacity</div>
            <span class="formula">Effective = Hours/Day × Efficiency %</span><br>
            = ${hoursPerDay} × ${(efficiency * 100).toFixed(0)}% = <strong>${effectiveHours.toFixed(1)} hrs/day</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 2: Total Available Capacity</div>
            <span class="formula">Total Capacity = Effective Hours × Days</span><br>
            = ${effectiveHours.toFixed(1)} × ${daysToShow} = <strong>${totalCapacity.toFixed(1)} hrs</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 3: Total Load</div>
            Total from ${capacityOrders.length} work orders = <strong>${totalLoad.toFixed(1)} hrs</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 4: Utilization</div>
            <span class="formula">Utilization = Total Load ÷ Total Capacity × 100</span><br>
            = ${totalLoad.toFixed(1)} ÷ ${totalCapacity.toFixed(1)} × 100 = <strong>${utilization.toFixed(0)}%</strong>
            ${overloadDays > 0 ? '<br><span style="color:#e74c3c;">⚠ ' + overloadDays + ' day(s) exceed capacity</span>' : ''}
        </div>
    `;

    // Chart
    const days = [];
    const loads = [];
    const colors = [];

    for (let d = 1; d <= daysToShow; d++) {
        days.push(`Day ${d}`);
        loads.push(loadByDay[d]?.total || 0);
        colors.push((loadByDay[d]?.total || 0) > effectiveHours ? '#e74c3c' : '#4a9f6e');
    }

    ForgeViz.render(document.getElementById('capacity-chart'), {
        title: '', chart_type: 'bar',
        traces: [
            { x: days, y: loads, name: 'Load', trace_type: 'bar', color: colors },
            { x: days, y: Array(daysToShow).fill(effectiveHours), name: 'Capacity', trace_type: 'line', color: '#e8c547', width: 2 }
        ],
        reference_lines: [
            { value: effectiveHours, axis: 'y', color: '#e8c547', dash: 'dashed', label: 'Capacity' }
        ],
        x_axis: { label: '' }, y_axis: { label: 'Hours' }
    });
}

// ============================================================================
// Mixed-Model Sequencer
// ============================================================================

let mixedProducts = [
    { id: 'A', name: 'Product A', quantity: 10, color: '#4a9f6e' },
    { id: 'B', name: 'Product B', quantity: 6, color: '#3498db' },
    { id: 'C', name: 'Product C', quantity: 4, color: '#e8c547' },
];

function renderMixedProducts() {
    const container = document.getElementById('mixed-products');
    if (!container) return;

    container.innerHTML = mixedProducts.map((p, i) => `
        <div class="yamazumi-station" style="gap: 8px;">
            <input type="text" value="${p.id}" style="width: 40px; text-align: center; padding: 8px;"
                   oninput="updateMixedProduct(${i}, 'id', this.value)" placeholder="ID">
            <input type="text" value="${p.name}" style="flex: 1; padding: 8px;"
                   oninput="updateMixedProduct(${i}, 'name', this.value)" placeholder="Name">
            <input type="number" value="${p.quantity}" style="width: 60px; text-align: right;"
                   oninput="updateMixedProduct(${i}, 'quantity', this.value)" title="Quantity" min="1">
            <span style="color: var(--text-dim); font-size: 11px;">units</span>
            <input type="color" value="${p.color}" style="width: 30px; height: 30px; border: none; cursor: pointer;"
                   oninput="updateMixedProduct(${i}, 'color', this.value)">
            <button class="yamazumi-station-remove" onclick="removeMixedProduct(${i})">&times;</button>
        </div>
    `).join('');

    calcMixedModel();
}

function addMixedProduct() {
    const id = String.fromCharCode(65 + mixedProducts.length);
    const hue = (mixedProducts.length * 137) % 360;
    mixedProducts.push({ id, name: `Product ${id}`, quantity: 5, color: `hsl(${hue}, 60%, 50%)` });
    renderMixedProducts();
}

function removeMixedProduct(idx) {
    mixedProducts.splice(idx, 1);
    renderMixedProducts();
}

function updateMixedProduct(idx, field, value) {
    if (field === 'quantity') value = parseInt(value) || 1;
    mixedProducts[idx][field] = value;
    calcMixedModel();
}

function pullFromHeijunka() {
    if (typeof heijunkaData !== 'undefined' && heijunkaData.length > 0) {
        mixedProducts = heijunkaData.map((h, i) => ({
            id: h.product,
            name: h.product,
            quantity: h.daily,
            color: `hsl(${i * 137 % 360}, 60%, 50%)`
        }));
        renderMixedProducts();
        showToast('Pulled from Heijunka');
    }
}

function calcMixedModel() {
    const method = document.querySelector('input[name="mix-method"]:checked')?.value || 'ratio';
    const container = document.getElementById('mixed-sequence');

    const totalQty = mixedProducts.reduce((sum, p) => sum + p.quantity, 0);

    let sequence = [];

    if (method === 'batch') {
        // Simple batched sequence
        mixedProducts.forEach(p => {
            for (let i = 0; i < p.quantity; i++) {
                sequence.push(p);
            }
        });
    } else if (method === 'ratio' || method === 'smooth') {
        // Goal chasing / ratio-based leveling
        const targets = mixedProducts.map(p => ({ ...p, target: 0, actual: 0 }));

        for (let i = 0; i < totalQty; i++) {
            // Update targets
            const progress = (i + 1) / totalQty;
            targets.forEach(t => {
                t.target = progress * t.quantity;
            });

            // Find product most behind target
            let bestIdx = 0;
            let maxDiff = -Infinity;
            targets.forEach((t, idx) => {
                const diff = t.target - t.actual;
                if (diff > maxDiff) {
                    maxDiff = diff;
                    bestIdx = idx;
                }
            });

            sequence.push(mixedProducts[bestIdx]);
            targets[bestIdx].actual++;
        }
    }

    // Render sequence
    if (container) {
        container.innerHTML = sequence.map(p => `
            <div style="width: 24px; height: 24px; background: ${p.color}; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 600; color: white;" title="${p.name}">${p.id}</div>
        `).join('');
    }

    // Calculate metrics
    let maxConsec = 1;
    let currentConsec = 1;
    for (let i = 1; i < sequence.length; i++) {
        if (sequence[i].id === sequence[i-1].id) {
            currentConsec++;
            maxConsec = Math.max(maxConsec, currentConsec);
        } else {
            currentConsec = 1;
        }
    }

    // Smoothness index (lower is better)
    let smoothness = 0;
    mixedProducts.forEach(p => {
        const idealGap = totalQty / p.quantity;
        const positions = sequence.map((s, i) => s.id === p.id ? i : -1).filter(i => i >= 0);
        for (let i = 1; i < positions.length; i++) {
            const gap = positions[i] - positions[i-1];
            smoothness += Math.abs(gap - idealGap);
        }
    });

    document.getElementById('mixed-length').textContent = `${totalQty} units`;
    document.getElementById('mixed-max-consec').textContent = maxConsec;
    document.getElementById('mixed-smoothness').textContent = smoothness.toFixed(1) + (method === 'batch' ? ' (batched)' : '');

    // Comparison chart
    const batchSeq = [];
    mixedProducts.forEach(p => {
        for (let i = 0; i < p.quantity; i++) batchSeq.push(p);
    });

    const leveledCumulative = {};
    const batchedCumulative = {};
    mixedProducts.forEach(p => {
        leveledCumulative[p.id] = [];
        batchedCumulative[p.id] = [];
    });

    let leveledCounts = {};
    let batchedCounts = {};
    mixedProducts.forEach(p => { leveledCounts[p.id] = 0; batchedCounts[p.id] = 0; });

    for (let i = 0; i < totalQty; i++) {
        leveledCounts[sequence[i].id]++;
        batchedCounts[batchSeq[i].id]++;
        mixedProducts.forEach(p => {
            leveledCumulative[p.id].push(leveledCounts[p.id]);
            batchedCumulative[p.id].push(batchedCounts[p.id]);
        });
    }

    const traces = [];
    mixedProducts.forEach(p => {
        traces.push({
            x: Array.from({ length: totalQty }, (_, i) => i + 1),
            y: leveledCumulative[p.id],
            type: 'scatter',
            mode: 'lines',
            name: `${p.id} (leveled)`,
            line: { color: p.color, width: 2 }
        });
        traces.push({
            x: Array.from({ length: totalQty }, (_, i) => i + 1),
            y: batchedCumulative[p.id],
            type: 'scatter',
            mode: 'lines',
            name: `${p.id} (batched)`,
            line: { color: p.color, width: 1, dash: 'dot' }
        });
    });

    ForgeViz.render(document.getElementById('mixed-chart'), {
        title: '', chart_type: 'line',
        traces: traces.map(t => ({
            x: t.x, y: t.y, name: t.name, trace_type: 'line',
            color: t.line.color, width: t.line.width || 2
        })),
        x_axis: { label: 'Sequence Position' }, y_axis: { label: 'Cumulative Count' }
    });
}

function pushMixedToLineSim() {
    const method = document.querySelector('input[name="mix-method"]:checked')?.value || 'ratio';

    // Generate sequence
    const totalQty = mixedProducts.reduce((sum, p) => sum + p.quantity, 0);
    const sequence = [];
    const targets = mixedProducts.map(p => ({ ...p, target: 0, actual: 0 }));

    for (let i = 0; i < totalQty; i++) {
        const progress = (i + 1) / totalQty;
        targets.forEach(t => { t.target = progress * t.quantity; });

        let bestIdx = 0;
        let maxDiff = -Infinity;
        targets.forEach((t, idx) => {
            const diff = t.target - t.actual;
            if (diff > maxDiff) { maxDiff = diff; bestIdx = idx; }
        });

        sequence.push(mixedProducts[bestIdx]);
        targets[bestIdx].actual++;
    }

    // Create products and orders in Line Sim
    if (typeof lineProducts !== 'undefined') {
        lineProducts = mixedProducts.map(p => ({
            id: p.id,
            name: p.name,
            color: p.color,
            ctMultiplier: 1.0
        }));
    }

    // Group consecutive same products into orders
    if (typeof lineOrders !== 'undefined') {
        lineOrders = [];
        let current = null;
        let count = 0;
        let orderId = 1;

        sequence.forEach((p, i) => {
            if (current === p.id) {
                count++;
            } else {
                if (current !== null) {
                    lineOrders.push({ id: orderId++, product: current, quantity: count, dueTime: 600 });
                }
                current = p.id;
                count = 1;
            }
        });
        if (current !== null) {
            lineOrders.push({ id: orderId, product: current, quantity: count, dueTime: 600 });
        }
    }

    showCalc('line-sim');
    setTimeout(() => {
        if (typeof renderProducts === 'function') renderProducts();
        if (typeof renderOrders === 'function') renderOrders();
        if (typeof updateSimMode === 'function') {
            document.getElementById('ls-sim-mode').value = 'orders';
            updateSimMode();
        }
        if (typeof initLineSim === 'function') initLineSim();
    }, 100);
    showToast('Pushed leveled sequence to Line Simulator');
}

function pullFromMixedModel() {
    if (typeof mixedProducts === 'undefined' || mixedProducts.length === 0) {
        showToast('No Mixed-Model data — configure products in Mixed-Model first', 'warning');
        return;
    }
    pushMixedToLineSim();
}

// --- Cross-Simulator Station Sharing ---

function pullLinesToKanban() {
    if (lineStations.length === 0) { showToast('No Line Sim stations — add stations or import from VSM first', 'warning'); return; }
    kanbanStations.length = 0;
    lineStations.forEach(s => kanbanStations.push({ name: s.name, cycleTime: s.cycleTime }));
    if (typeof renderKanbanStations === 'function') renderKanbanStations();
    if (typeof resetKanbanSim === 'function') resetKanbanSim();
    showToast(`Pulled ${kanbanStations.length} stations from Line Sim`);
}

function pullKanbanToLine() {
    if (kanbanStations.length === 0) { showToast('No Kanban Sim stations — add stations first', 'warning'); return; }
    lineStations.length = 0;
    kanbanStations.forEach(s => lineStations.push({ name: s.name, cycleTime: s.cycleTime }));
    if (typeof renderLineStations === 'function') renderLineStations();
    if (typeof resetLineSim === 'function') resetLineSim();
    showToast(`Pulled ${lineStations.length} stations from Kanban Sim`);
}

function pullLinesToTOC() {
    if (lineStations.length === 0) { showToast('No Line Sim stations — add stations or import from VSM first', 'warning'); return; }
    tocStationsData.length = 0;
    lineStations.forEach(s => {
        const capacity = Math.round(3600 / (s.cycleTime || 60));
        tocStationsData.push({ name: s.name, capacity });
    });
    if (typeof renderTocStations === 'function') renderTocStations();
    if (typeof resetTocSim === 'function') resetTocSim();
    showToast(`Pulled ${tocStationsData.length} stations from Line Sim (cycle time → capacity)`);
}

function pullBottleneckToTOC() {
    if (bottleneckData.length === 0) { showToast('No Bottleneck data — add process steps first', 'warning'); return; }
    tocStationsData.length = 0;
    bottleneckData.forEach(s => {
        const capacity = Math.round(3600 / (s.time || 60));
        tocStationsData.push({ name: s.name, capacity });
    });
    if (typeof renderTocStations === 'function') renderTocStations();
    if (typeof resetTocSim === 'function') resetTocSim();
    showToast(`Pulled ${tocStationsData.length} stations from Bottleneck (cycle time → capacity)`);
}

// ============================================================================
// Due Date Risk Simulator
// ============================================================================

let ddsOrders = [
    { id: 1, name: 'Order A', processTime: 120, dueDate: 480 },
    { id: 2, name: 'Order B', processTime: 90, dueDate: 360 },
    { id: 3, name: 'Order C', processTime: 150, dueDate: 600 },
];

function renderDDSOrders() {
    const container = document.getElementById('dds-orders');
    if (!container) return;

    container.innerHTML = ddsOrders.map((order, i) => `
        <div class="yamazumi-station" style="gap: 8px;">
            <input type="text" value="${order.name}" style="flex: 1; padding: 8px;"
                   oninput="updateDDSOrder(${i}, 'name', this.value)" placeholder="Order name">
            <input type="number" value="${order.processTime}" style="width: 70px; text-align: right;"
                   oninput="updateDDSOrder(${i}, 'processTime', this.value)" title="Process time (min)">
            <span style="color: var(--text-dim); font-size: 11px;">min</span>
            <input type="number" value="${order.dueDate}" style="width: 80px; text-align: right;"
                   oninput="updateDDSOrder(${i}, 'dueDate', this.value)" title="Due date (min from now)">
            <span style="color: var(--text-dim); font-size: 11px;">due</span>
            <button class="yamazumi-station-remove" onclick="removeDDSOrder(${i})">&times;</button>
        </div>
    `).join('');
}

function addDDSOrder() {
    const id = ddsOrders.length > 0 ? Math.max(...ddsOrders.map(o => o.id)) + 1 : 1;
    ddsOrders.push({ id, name: `Order ${String.fromCharCode(64 + id)}`, processTime: 60, dueDate: 480 });
    renderDDSOrders();
}

function removeDDSOrder(idx) {
    ddsOrders.splice(idx, 1);
    renderDDSOrders();
}

function updateDDSOrder(idx, field, value) {
    if (field === 'processTime' || field === 'dueDate') value = parseFloat(value) || 0;
    ddsOrders[idx][field] = value;
}

async function ddsReschedule(idx) {
    const order = ddsOrders[idx];
    const newDue = await svendPrompt(`Reschedule ${order.name}\nCurrent due: ${order.dueDate} min\nSuggested: ${Math.round(order.dueDate * 1.3)} min\n\nEnter new due date (minutes):`, Math.round(order.dueDate * 1.3));
    if (newDue !== null) {
        ddsOrders[idx].dueDate = parseFloat(newDue) || order.dueDate;
        renderDDSOrders();
        runDueDateSim();
        showToast(`${order.name} rescheduled to ${newDue} min`);
    }
}

function ddsFlag(idx) {
    const order = ddsOrders[idx];
    order.flagged = true;
    showToast(`${order.name} flagged for expediting`);
    runDueDateSim();
}

function loadSampleDDSOrders() {
    ddsOrders = [
        { id: 1, name: 'Rush Order', processTime: 60, dueDate: 180 },
        { id: 2, name: 'Standard A', processTime: 120, dueDate: 480 },
        { id: 3, name: 'Standard B', processTime: 90, dueDate: 400 },
        { id: 4, name: 'Large Job', processTime: 200, dueDate: 720 },
        { id: 5, name: 'Quick Turn', processTime: 45, dueDate: 300 },
    ];
    renderDDSOrders();
}

function pullFromSequencerToDDS() {
    ddsOrders = sequencerJobs.map((job, i) => ({
        id: i + 1,
        name: job.name,
        processTime: job.processTime,
        dueDate: job.dueDate
    }));
    renderDDSOrders();
    showToast('Pulled from Job Sequencer');
}

function runDueDateSim() {
    if (ddsOrders.length === 0) return;

    const cv = parseFloat(document.getElementById('dds-cv')?.value) || 0.15;
    const breakdownProb = (parseFloat(document.getElementById('dds-breakdown')?.value) || 5) / 100;
    const breakdownDur = parseFloat(document.getElementById('dds-breakdown-dur')?.value) || 60;
    const runs = parseInt(document.getElementById('dds-runs')?.value) || 500;

    // Results storage
    const completionTimes = ddsOrders.map(() => []);
    let totalOnTime = 0;

    // Monte Carlo simulation
    for (let run = 0; run < runs; run++) {
        let currentTime = 0;
        let runOnTime = true;

        ddsOrders.forEach((order, i) => {
            // Process time with variation
            const stdDev = order.processTime * cv;
            const actualProcess = Math.max(1, order.processTime + gaussianRandom() * stdDev);

            // Random breakdown
            const hasBreakdown = Math.random() < breakdownProb;
            const breakdownTime = hasBreakdown ? breakdownDur * (0.5 + Math.random()) : 0;

            currentTime += actualProcess + breakdownTime;
            completionTimes[i].push(currentTime);

            if (currentTime > order.dueDate) {
                runOnTime = false;
            }
        });

        if (runOnTime) totalOnTime++;
    }

    // Calculate statistics
    const overallOTD = (totalOnTime / runs) * 100;

    const orderResults = ddsOrders.map((order, i) => {
        const times = completionTimes[i].sort((a, b) => a - b);
        const mean = times.reduce((a, b) => a + b, 0) / times.length;
        const p50 = times[Math.floor(times.length * 0.5)];
        const p95 = times[Math.floor(times.length * 0.95)];
        const onTimeCount = times.filter(t => t <= order.dueDate).length;
        const onTimePct = (onTimeCount / runs) * 100;
        const avgDelta = mean - order.dueDate;

        return {
            name: order.name,
            dueDate: order.dueDate,
            mean,
            p50,
            p95,
            onTimePct,
            avgDelta
        };
    });

    // Update display
    const otdColor = overallOTD >= 95 ? '#4a9f6e' : overallOTD >= 80 ? '#e8c547' : '#e74c3c';
    document.getElementById('dds-otd').innerHTML = `<span style="color: ${otdColor}">${overallOTD.toFixed(1)}</span><span class="calc-result-unit">%</span>`;

    const avgDeltaAll = orderResults.reduce((sum, r) => sum + r.avgDelta, 0) / orderResults.length;
    document.getElementById('dds-avg-delta').innerHTML = avgDeltaAll > 0 ?
        `<span style="color: #e74c3c;">${avgDeltaAll.toFixed(0)} min late</span>` :
        `<span style="color: #4a9f6e;">${Math.abs(avgDeltaAll).toFixed(0)} min early</span>`;

    const worstP95 = Math.max(...orderResults.map(r => r.p95 - r.dueDate));
    document.getElementById('dds-worst').innerHTML = worstP95 > 0 ?
        `<span style="color: #e74c3c;">${worstP95.toFixed(0)} min late</span>` :
        `<span style="color: #4a9f6e;">All on time</span>`;

    // Per-order results
    const resultsContainer = document.getElementById('dds-order-results');
    resultsContainer.innerHTML = orderResults.map((r, idx) => {
        const riskColor = r.onTimePct >= 95 ? '#4a9f6e' : r.onTimePct >= 80 ? '#e8c547' : '#e74c3c';
        const isHighRisk = r.onTimePct < 80;
        return `
            <div style="display: flex; align-items: center; gap: 12px; padding: 12px; background: var(--bg-secondary); border-radius: 8px; border-left: 3px solid ${riskColor};">
                <div style="flex: 1;">
                    <div style="font-weight: 500;">${r.name}${r.flagged ? ' <span style="color:#e8c547;">&#9873;</span>' : ''}</div>
                    <div style="font-size: 11px; color: var(--text-dim);">Due: ${r.dueDate} min | Avg: ${r.mean.toFixed(0)} min</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 18px; font-weight: 600; color: ${riskColor};">${r.onTimePct.toFixed(0)}%</div>
                    <div style="font-size: 10px; color: var(--text-dim);">on-time</div>
                </div>
                ${isHighRisk ? `
                <div style="display: flex; gap: 4px;">
                    <button onclick="ddsReschedule(${idx})" style="padding: 4px 8px; font-size: 11px; background: var(--accent); border: none; border-radius: 4px; color: white; cursor: pointer;" title="Adjust due date and re-run">Reschedule</button>
                    <button onclick="ddsFlag(${idx})" style="padding: 4px 8px; font-size: 11px; background: #e8c547; border: none; border-radius: 4px; color: #1a1d23; cursor: pointer;" title="Flag for expediting">Flag</button>
                </div>
                ` : ''}
            </div>
        `;
    }).join('');

    // Histogram of final order completion
    const finalTimes = completionTimes[completionTimes.length - 1];
    const minT = Math.min(...finalTimes);
    const maxT = Math.max(...finalTimes);
    const binCount = 20;
    const binWidth = (maxT - minT) / binCount;
    const bins = Array(binCount).fill(0);
    const binLabels = [];

    for (let i = 0; i < binCount; i++) {
        binLabels.push(Math.round(minT + i * binWidth));
    }

    finalTimes.forEach(t => {
        const bin = Math.min(binCount - 1, Math.floor((t - minT) / binWidth));
        bins[bin]++;
    });

    const lastDue = ddsOrders[ddsOrders.length - 1].dueDate;

    ForgeViz.render(document.getElementById('dds-chart'), {
        title: '', chart_type: 'bar',
        traces: [
            { x: binLabels, y: bins, name: 'Completion', trace_type: 'bar', color: binLabels.map(t => t <= lastDue ? '#4a9f6e' : '#e74c3c') }
        ],
        reference_lines: [
            { value: lastDue, axis: 'x', color: '#e8c547', dash: 'dashed', label: 'Due' }
        ],
        x_axis: { label: 'Completion Time (min)' }, y_axis: { label: 'Frequency' }
    });
}

// Gaussian random using Box-Muller
function gaussianRandom() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
