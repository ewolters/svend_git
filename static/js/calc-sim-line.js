/**
 * calc-sim-line.js — Line Simulator for Operations Workbench
 *
 * Load order: after svend-math.js, svend-charts.js, svend-sim-core.js
 * Extracted from: calculators.html (inline script)
 *
 * Provides: Line flow simulator with order-driven mode, station stats,
 * bottleneck detection, A/B comparison, scenario management, VSM export.
 */

let lineStations = [
    { name: 'Station 1', cycleTime: 45 },
    { name: 'Station 2', cycleTime: 50 },
    { name: 'Station 3', cycleTime: 55 },
    { name: 'Station 4', cycleTime: 48 },
];

// Product types for order-driven mode
let lineProducts = [
    { id: 'A', name: 'Product A', color: '#4a9f6e', ctMultiplier: 1.0 },
    { id: 'B', name: 'Product B', color: '#3498db', ctMultiplier: 1.2 },
];

// Order queue for order-driven mode
let lineOrders = [];
let orderIdCounter = 1;

let lineSimState = {
    running: false,
    interval: null,
    time: 0,
    completed: 0,
    history: { time: [], wip: [], throughput: [] },
    stationStats: [], // { working: 0, blocked: 0, starved: 0, processed: 0 }
    // Order-driven mode state
    currentOrder: null,
    currentOrderProgress: 0,
    currentProduct: null,
    changingOver: false,
    changeoverRemaining: 0,
    totalChangeoverTime: 0,
    orderResults: [], // { orderId, product, qty, dueTime, completedTime, wasLate, lateReason }
};

function renderLineStations() {
    const container = document.getElementById('ls-stations');
    container.innerHTML = lineStations.map((s, i) => `
        <div style="display: flex; align-items: center; gap: 12px; padding: 8px 12px; background: var(--bg-secondary); border-radius: 6px;">
            <input type="text" value="${s.name}" onchange="lineStations[${i}].name = this.value; renderLineVisual();"
                   style="width: 100px; padding: 6px 8px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary); font-size: 13px;">
            <span style="color: var(--text-dim); font-size: 12px;">CT:</span>
            <input type="number" value="${s.cycleTime}" onchange="lineStations[${i}].cycleTime = parseFloat(this.value) || 30; renderLineVisual();"
                   style="width: 60px; padding: 6px 8px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary); font-size: 13px;">
            <span style="color: var(--text-dim); font-size: 12px;">sec</span>
            <button onclick="removeLineStation(${i})" style="padding: 4px 8px; background: transparent; border: none; color: var(--text-dim); cursor: pointer; font-size: 16px;">&times;</button>
        </div>
    `).join('');
    renderLineVisual();
}

function addLineStation() {
    lineStations.push({ name: `Station ${lineStations.length + 1}`, cycleTime: 45 });
    renderLineStations();
}

function removeLineStation(idx) {
    if (lineStations.length > 2) {
        lineStations.splice(idx, 1);
        renderLineStations();
    }
}

function updateLineSimParams() {
    const cov = parseFloat(document.getElementById('ls-cov').value);
    const labels = ['0% — Perfect automation', '5% — High automation', '10% — Low variability', '15% — Typical manual', '25% — High variability', '35% — Problematic', '50% — Chaos'];
    const label = cov <= 0 ? labels[0] : cov <= 5 ? labels[1] : cov <= 10 ? labels[2] : cov <= 15 ? labels[3] : cov <= 25 ? labels[4] : cov <= 35 ? labels[5] : labels[6];
    document.getElementById('ls-cov-label').textContent = label;

    // Handle flow mode
    const flowMode = document.getElementById('ls-flow-mode').value;
    const batchContainer = document.getElementById('ls-batch-size-container');
    const flowHint = document.getElementById('ls-flow-hint');

    if (flowMode === 'batch') {
        batchContainer.style.display = 'block';
        const batchSize = document.getElementById('ls-batch-size').value;
        flowHint.textContent = `Batches of ${batchSize} units — watch WIP accumulate`;
        flowHint.style.color = '#f39c12';
    } else {
        batchContainer.style.display = 'none';
        flowHint.textContent = 'Units move individually through the line';
        flowHint.style.color = '';
    }

    renderLineVisual();
}

function renderLineVisual() {
    const takt = parseFloat(document.getElementById('ls-takt').value) || 60;
    const maxWip = parseInt(document.getElementById('ls-max-wip').value) || 5;
    const container = document.getElementById('ls-line');
    const simMode = document.getElementById('ls-sim-mode').value;

    // Changeover indicator
    const isChangingOver = lineSimState.changingOver;
    const changeoverRemaining = lineSimState.changeoverRemaining || 0;

    // Current order/product info
    const currentOrder = lineSimState.currentOrder;
    const currentProduct = lineSimState.currentProduct;
    const orderProgress = lineSimState.currentOrderProgress || 0;

    // Build visual: [Input/Order] -> [Station 1] -> [Buffer] -> [Station 2] -> ... -> [Output]
    let inputLabel = 'Input';
    let inputColor = 'linear-gradient(135deg, #4a9f6e 0%, #3d8a5e 100%)';
    let inputContent = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>';

    if (simMode === 'orders') {
        if (isChangingOver) {
            inputLabel = `C/O ${changeoverRemaining.toFixed(0)}s`;
            inputColor = 'linear-gradient(135deg, #f39c12 0%, #e67e22 100%)';
            inputContent = '<span style="color: white; font-size: 16px;">⟳</span>';
        } else if (currentOrder) {
            const productColor = currentProduct?.color || '#4a9f6e';
            inputLabel = `#${currentOrder.id}: ${orderProgress}/${currentOrder.qty}`;
            inputColor = `linear-gradient(135deg, ${productColor} 0%, ${productColor}dd 100%)`;
            inputContent = `<span style="color: white; font-weight: 600; font-size: 12px;">${currentProduct?.name?.charAt(0) || '?'}</span>`;
        }
    }

    let html = `
        <div style="display: flex; flex-direction: column; align-items: center; gap: 4px;">
            <div style="width: 50px; height: 50px; background: ${inputColor}; border-radius: 50%; display: flex; align-items: center; justify-content: center; ${isChangingOver ? 'animation: pulse 1s infinite;' : ''}">
                ${inputContent}
            </div>
            <span style="font-size: 10px; color: var(--text-dim); max-width: 70px; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${inputLabel}</span>
        </div>
    `;

    lineStations.forEach((station, i) => {
        const utilization = (station.cycleTime / takt) * 100;
        const overloaded = utilization > 100;
        const stationColor = overloaded ? '#e74c3c' : utilization > 85 ? '#f39c12' : '#4a9f6e';
        const state = lineSimState.stationStats[i] || {};
        const stateLabel = state.currentState || 'idle';
        const stateColor = stateLabel === 'working' ? stationColor : stateLabel === 'blocked' ? '#e74c3c' : stateLabel === 'starved' ? '#f39c12' : 'var(--text-dim)';

        // Arrow
        html += `<svg width="30" height="20" viewBox="0 0 30 20" style="flex-shrink: 0;"><path d="M0 10 L20 10 M15 5 L20 10 L15 15" stroke="var(--border)" stroke-width="2" fill="none"/></svg>`;

        // Station
        html += `
            <div style="display: flex; flex-direction: column; align-items: center; gap: 4px;">
                <div id="ls-station-${i}" style="width: 80px; height: 60px; background: var(--bg-card); border: 2px solid ${stationColor}; border-radius: 8px; display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative;">
                    <span style="font-size: 11px; font-weight: 600; color: var(--text-primary);">${station.name}</span>
                    <span style="font-size: 10px; color: ${stationColor};">${station.cycleTime}s</span>
                    <div id="ls-station-state-${i}" style="position: absolute; top: -8px; right: -8px; width: 16px; height: 16px; background: ${stateColor}; border-radius: 50%; border: 2px solid var(--bg-secondary);"></div>
                </div>
                <span style="font-size: 10px; color: var(--text-dim);">${utilization.toFixed(0)}%</span>
            </div>
        `;

        // WIP Buffer (except after last station)
        if (i < lineStations.length - 1) {
            const wip = state.outputBuffer || 0;
            const wipPct = (wip / maxWip) * 100;
            const wipColor = wipPct >= 100 ? '#e74c3c' : wipPct > 60 ? '#f39c12' : '#4a9f6e';

            html += `<svg width="30" height="20" viewBox="0 0 30 20" style="flex-shrink: 0;"><path d="M0 10 L20 10 M15 5 L20 10 L15 15" stroke="var(--border)" stroke-width="2" fill="none"/></svg>`;
            html += `
                <div style="display: flex; flex-direction: column; align-items: center; gap: 4px;">
                    <div id="ls-buffer-${i}" style="width: 40px; height: 50px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; display: flex; flex-direction: column; justify-content: flex-end; overflow: hidden;">
                        <div style="height: ${wipPct}%; background: ${wipColor}; transition: height 0.3s;"></div>
                    </div>
                    <span id="ls-buffer-label-${i}" style="font-size: 10px; color: var(--text-dim);">${wip}/${maxWip}</span>
                </div>
            `;
        }
    });

    // Output
    html += `
        <svg width="30" height="20" viewBox="0 0 30 20" style="flex-shrink: 0;"><path d="M0 10 L20 10 M15 5 L20 10 L15 15" stroke="var(--border)" stroke-width="2" fill="none"/></svg>
        <div style="display: flex; flex-direction: column; align-items: center; gap: 4px;">
            <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                <span id="ls-output-count" style="color: white; font-weight: 600; font-size: 14px;">${lineSimState.completed}</span>
            </div>
            <span style="font-size: 10px; color: var(--text-dim);">Output</span>
        </div>
    `;

    container.innerHTML = html;
}

function startLineSim() {
    if (lineSimState.running) {
        // Pause
        lineSimState.running = false;
        clearInterval(lineSimState.interval);
        document.getElementById('ls-start').innerHTML = '▶ Resume';
        return;
    }

    lineSimState.running = true;
    document.getElementById('ls-start').innerHTML = '⏸ Pause';

    const takt = parseFloat(document.getElementById('ls-takt').value) || 60;
    const cov = parseFloat(document.getElementById('ls-cov').value) / 100;
    const maxWip = parseInt(document.getElementById('ls-max-wip').value) || 5;
    const speedFactor = parseInt(document.getElementById('ls-speed').value);
    const flowMode = document.getElementById('ls-flow-mode').value;
    const batchSize = flowMode === 'batch' ? parseInt(document.getElementById('ls-batch-size').value) || 5 : 1;
    const simMode = document.getElementById('ls-sim-mode').value;
    const changeoverTime = parseFloat(document.getElementById('ls-changeover-time')?.value) || 120;

    // Initialize station states if fresh start
    if (lineSimState.time === 0) {
        lineSimState.stationStats = lineStations.map(() => ({
            working: 0,
            blocked: 0,
            starved: 0,
            processed: 0,
            currentState: 'starved',
            timeRemaining: 0,
            outputBuffer: 0,
            hasUnit: false,
            batchCount: 0,
        }));
        lineSimState.history = { time: [], wip: [], throughput: [] };
        lineSimState.completed = 0;
        lineSimState.batchSize = batchSize;
        lineSimState.currentOrder = null;
        lineSimState.currentOrderProgress = 0;
        lineSimState.currentProduct = null;
        lineSimState.changingOver = false;
        lineSimState.changeoverRemaining = 0;
        lineSimState.totalChangeoverTime = 0;
        lineSimState.orderResults = [];

        // In order mode, mark first order as in-progress
        if (simMode === 'orders' && lineOrders.length > 0) {
            lineOrders.forEach(o => { o.status = 'pending'; });
        }
    }

    // Simulation tick (100ms real time = speedFactor seconds sim time)
    const tickInterval = 100;
    const simSecondsPerTick = speedFactor * 0.5;

    lineSimState.interval = setInterval(() => {
        lineSimState.time += simSecondsPerTick;

        const breakdownsEnabled = document.getElementById('ls-breakdowns').checked;

        // === ORDER-DRIVEN MODE: Handle changeovers and order tracking ===
        let ctMultiplier = 1.0;
        if (simMode === 'orders') {
            // Handle changeover state
            if (lineSimState.changingOver) {
                lineSimState.changeoverRemaining -= simSecondsPerTick;
                lineSimState.totalChangeoverTime += simSecondsPerTick;
                if (lineSimState.changeoverRemaining <= 0) {
                    lineSimState.changingOver = false;
                }
                // During changeover, all stations are idle - skip processing
                updateOrderMetrics(simSecondsPerTick);
                return;
            }

            // Get current order or next pending order
            if (!lineSimState.currentOrder) {
                const nextOrder = lineOrders.find(o => o.status === 'pending');
                if (nextOrder) {
                    // Check if we need a changeover
                    const newProduct = lineProducts[nextOrder.productIdx];
                    if (lineSimState.currentProduct && lineSimState.currentProduct.id !== newProduct?.id) {
                        // Trigger changeover
                        lineSimState.changingOver = true;
                        lineSimState.changeoverRemaining = changeoverTime;
                        // Record if this changeover will cause lateness
                        if (lineSimState.time + changeoverTime > nextOrder.dueTime) {
                            nextOrder.lateReason = `Changeover to ${newProduct?.name} started at t=${lineSimState.time.toFixed(0)}s, will take ${changeoverTime}s`;
                        }
                        updateOrderMetrics(simSecondsPerTick);
                        return;
                    }

                    // Start the order
                    lineSimState.currentOrder = nextOrder;
                    lineSimState.currentOrderProgress = 0;
                    lineSimState.currentProduct = newProduct;
                    nextOrder.status = 'in-progress';
                    nextOrder.startTime = lineSimState.time;
                    renderOrders();
                } else {
                    // No more orders - stop simulation
                    lineSimState.running = false;
                    clearInterval(lineSimState.interval);
                    document.getElementById('ls-start').innerHTML = '▶ All Orders Complete';
                    updateDeliveryMetrics();
                    return;
                }
            }

            // Apply product cycle time multiplier
            ctMultiplier = lineSimState.currentProduct?.ctMultiplier || 1.0;
        }

        // Process each station (from last to first to avoid double-moving)
        for (let i = lineStations.length - 1; i >= 0; i--) {
            const station = lineStations[i];
            const state = lineSimState.stationStats[i];
            const prevState = i > 0 ? lineSimState.stationStats[i - 1] : null;
            const isFirstStation = i === 0;
            const isLastStation = i === lineStations.length - 1;

            // Check for breakdown (if enabled)
            if (breakdownsEnabled && state.currentState !== 'down') {
                // ~0.5% chance per tick of breakdown (MTBF ~100 ticks ≈ 50 sim seconds)
                if (Math.random() < 0.005) {
                    state.currentState = 'down';
                    state.downTimeRemaining = 15 + Math.random() * 30; // 15-45 sec repair
                    state.downtime = (state.downtime || 0);
                    // Record if breakdown will cause lateness
                    if (simMode === 'orders' && lineSimState.currentOrder) {
                        const order = lineSimState.currentOrder;
                        if (!order.lateReason && lineSimState.time > order.dueTime * 0.7) {
                            order.lateReason = `Breakdown at ${station.name} (t=${lineSimState.time.toFixed(0)}s)`;
                        }
                    }
                }
            }

            // Handle breakdown state
            if (state.currentState === 'down') {
                state.downTimeRemaining -= simSecondsPerTick;
                state.downtime = (state.downtime || 0) + simSecondsPerTick;
                if (state.downTimeRemaining <= 0) {
                    state.currentState = state.hasUnit ? 'working' : 'starved';
                }
                continue; // Skip normal processing while down
            }

            // Add variability to cycle time (per unit in batch), apply product multiplier
            const actualCT = station.cycleTime * ctMultiplier * (1 + (Math.random() - 0.5) * 2 * cov);

            if (state.hasUnit) {
                // Currently processing
                state.timeRemaining -= simSecondsPerTick;

                if (state.timeRemaining <= 0) {
                    // Finished processing
                    if (isLastStation) {
                        // Output the unit(s)
                        const unitsCompleted = state.batchCount;
                        lineSimState.completed += unitsCompleted;
                        state.hasUnit = false;
                        state.processed += unitsCompleted;
                        state.batchCount = 0;
                        state.currentState = 'starved';

                        // Order mode: track progress
                        if (simMode === 'orders' && lineSimState.currentOrder) {
                            lineSimState.currentOrderProgress += unitsCompleted;
                            if (lineSimState.currentOrderProgress >= lineSimState.currentOrder.qty) {
                                // Order complete!
                                const order = lineSimState.currentOrder;
                                order.status = 'complete';
                                order.completedTime = lineSimState.time;
                                order.wasLate = lineSimState.time > order.dueTime;
                                if (order.wasLate && !order.lateReason) {
                                    order.lateReason = 'Cumulative delays exceeded buffer';
                                }

                                lineSimState.orderResults.push({
                                    orderId: order.id,
                                    productIdx: order.productIdx,
                                    qty: order.qty,
                                    dueTime: order.dueTime,
                                    completedTime: order.completedTime,
                                    wasLate: order.wasLate,
                                    lateReason: order.lateReason,
                                });

                                lineSimState.currentOrder = null;
                                lineSimState.currentOrderProgress = 0;
                                renderOrders();
                                updateDeliveryMetrics();
                            }
                        }
                    } else if (state.outputBuffer + state.batchCount <= maxWip) {
                        // Move to buffer
                        state.outputBuffer += state.batchCount;
                        state.hasUnit = false;
                        state.processed += state.batchCount;
                        state.batchCount = 0;
                        state.currentState = 'starved';
                    } else {
                        // Blocked!
                        state.currentState = 'blocked';
                        // Record blocking as potential late cause
                        if (simMode === 'orders' && lineSimState.currentOrder) {
                            const order = lineSimState.currentOrder;
                            if (!order.lateReason && lineSimState.time > order.dueTime * 0.8) {
                                order.lateReason = `Blocking at ${station.name} (buffer full, t=${lineSimState.time.toFixed(0)}s)`;
                            }
                        }
                        state.blocked += simSecondsPerTick;
                    }
                } else {
                    state.currentState = 'working';
                    state.working += simSecondsPerTick;
                }
            } else {
                // Try to get a unit (or batch)
                if (isFirstStation) {
                    // Order mode: only pull if we have an active order with remaining qty
                    if (simMode === 'orders') {
                        if (lineSimState.currentOrder && lineSimState.currentOrderProgress < lineSimState.currentOrder.qty) {
                            const remaining = lineSimState.currentOrder.qty - lineSimState.currentOrderProgress;
                            const pullQty = Math.min(batchSize, remaining);
                            state.hasUnit = true;
                            state.batchCount = pullQty;
                            state.timeRemaining = actualCT * pullQty;
                            state.currentState = 'working';
                        } else {
                            // No order or order complete - starve until next order
                            state.currentState = 'starved';
                            state.starved += simSecondsPerTick;
                        }
                    } else {
                        // Infinite supply mode
                        state.hasUnit = true;
                        state.batchCount = batchSize;
                        state.timeRemaining = actualCT * batchSize;
                        state.currentState = 'working';
                    }
                } else if (prevState && prevState.outputBuffer >= batchSize) {
                    // Pull batch from previous buffer
                    prevState.outputBuffer -= batchSize;
                    state.hasUnit = true;
                    state.batchCount = batchSize;
                    state.timeRemaining = actualCT * batchSize;
                    state.currentState = 'working';
                } else if (batchSize === 1 && prevState && prevState.outputBuffer > 0) {
                    // One-piece flow: pull single unit
                    prevState.outputBuffer--;
                    state.hasUnit = true;
                    state.batchCount = 1;
                    state.timeRemaining = actualCT;
                    state.currentState = 'working';
                } else {
                    // Starved (waiting for batch to accumulate)
                    state.currentState = 'starved';
                    state.starved += simSecondsPerTick;
                }
            }
        }

        // Update metrics
        const totalWip = lineSimState.stationStats.reduce((sum, s) => sum + s.outputBuffer + (s.hasUnit ? s.batchCount : 0), 0);
        const throughputPerHour = (lineSimState.completed / lineSimState.time) * 3600;

        // Calculate line efficiency (excluding downtime from denominator for OEE-style calc)
        const totalWorking = lineSimState.stationStats.reduce((sum, s) => sum + s.working, 0);
        const totalTime = lineSimState.stationStats.reduce((sum, s) => sum + s.working + s.blocked + s.starved + (s.downtime || 0), 0);
        const efficiency = totalTime > 0 ? (totalWorking / totalTime) * 100 : 0;

        // Record history (every ~5 sim seconds)
        if (lineSimState.history.time.length === 0 || lineSimState.time - lineSimState.history.time[lineSimState.history.time.length - 1] >= 5) {
            lineSimState.history.time.push(lineSimState.time);
            lineSimState.history.wip.push(totalWip);
            lineSimState.history.throughput.push(throughputPerHour);
        }

        // Update UI
        document.getElementById('ls-completed').textContent = lineSimState.completed;
        document.getElementById('ls-throughput').innerHTML = `${throughputPerHour.toFixed(1)}<span class="calc-result-unit">/hr</span>`;
        document.getElementById('ls-total-wip').textContent = totalWip;
        document.getElementById('ls-efficiency').innerHTML = `${efficiency.toFixed(1)}<span class="calc-result-unit">%</span>`;

        // Publish live sim values
        SvendOps.publish('simThroughput', parseFloat(throughputPerHour.toFixed(1)), 'units/hr', 'Line Sim');
        SvendOps.publish('simWIP', totalWip, 'units', 'Line Sim');
        SvendOps.publish('simEfficiency', parseFloat(efficiency.toFixed(1)), '%', 'Line Sim');

        // Update visual
        updateLineVisual(maxWip);

        // Update chart
        updateLineChart();

        // Update station stats table
        updateLineStationStats(takt);

        // Bottleneck analysis
        updateLineBottleneck(takt);

    }, tickInterval);
}

function updateLineVisual(maxWip) {
    lineSimState.stationStats.forEach((state, i) => {
        const stateEl = document.getElementById(`ls-station-state-${i}`);
        if (stateEl) {
            const stateColor = state.currentState === 'down' ? '#666' :
                               state.currentState === 'working' ? '#4a9f6e' :
                               state.currentState === 'blocked' ? '#e74c3c' : '#f39c12';
            stateEl.style.background = stateColor;

            // Flash the station border on breakdown
            const stationEl = document.getElementById(`ls-station-${i}`);
            if (stationEl) {
                stationEl.style.borderColor = state.currentState === 'down' ? '#666' : '';
                stationEl.style.opacity = state.currentState === 'down' ? '0.6' : '1';
            }
        }

        if (i < lineStations.length - 1) {
            const bufferEl = document.getElementById(`ls-buffer-${i}`);
            const labelEl = document.getElementById(`ls-buffer-label-${i}`);
            if (bufferEl && labelEl) {
                const wipPct = (state.outputBuffer / maxWip) * 100;
                const wipColor = wipPct >= 100 ? '#e74c3c' : wipPct > 60 ? '#f39c12' : '#4a9f6e';
                bufferEl.innerHTML = `<div style="height: ${wipPct}%; background: ${wipColor}; transition: height 0.2s;"></div>`;
                labelEl.textContent = `${state.outputBuffer}/${maxWip}`;
            }
        }
    });

    const outputEl = document.getElementById('ls-output-count');
    if (outputEl) outputEl.textContent = lineSimState.completed;
}

function updateLineChart() {
    const h = lineSimState.history;
    if (h.time.length < 2) return;

    ForgeViz.render(document.getElementById('ls-chart'), {
        title: '', chart_type: 'line',
        traces: [
            { x: h.time, y: h.wip, name: 'WIP', trace_type: 'line', color: '#f39c12', width: 2 },
            { x: h.time, y: h.throughput, name: 'Throughput/hr', trace_type: 'line', color: '#4a9f6e', width: 2 }
        ],
        reference_lines: [], zones: [], markers: [],
        x_axis: { label: 'Time (sec)' }, y_axis: { label: 'WIP / Units/hr' }
    });
}

function updateLineStationStats(takt) {
    const container = document.getElementById('ls-station-stats');
    const breakdownsEnabled = document.getElementById('ls-breakdowns').checked;

    let html = `
        <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
            <thead>
                <tr style="border-bottom: 1px solid var(--border);">
                    <th style="padding: 8px; text-align: left;">Station</th>
                    <th style="padding: 8px; text-align: right;">CT</th>
                    <th style="padding: 8px; text-align: right;">Processed</th>
                    <th style="padding: 8px; text-align: right;">Working %</th>
                    <th style="padding: 8px; text-align: right;">Blocked %</th>
                    <th style="padding: 8px; text-align: right;">Starved %</th>
                    ${breakdownsEnabled ? '<th style="padding: 8px; text-align: right;">Down %</th>' : ''}
                    <th style="padding: 8px; text-align: left;">Status</th>
                </tr>
            </thead>
            <tbody>
    `;

    lineStations.forEach((station, i) => {
        const state = lineSimState.stationStats[i];
        const downtime = state.downtime || 0;
        const totalTime = state.working + state.blocked + state.starved + downtime;
        const workPct = totalTime > 0 ? (state.working / totalTime) * 100 : 0;
        const blockPct = totalTime > 0 ? (state.blocked / totalTime) * 100 : 0;
        const starvePct = totalTime > 0 ? (state.starved / totalTime) * 100 : 0;
        const downPct = totalTime > 0 ? (downtime / totalTime) * 100 : 0;

        const utilization = (station.cycleTime / takt) * 100;
        let status = 'Normal';
        let statusColor = '#4a9f6e';
        if (state.currentState === 'down') { status = 'DOWN'; statusColor = '#666'; }
        else if (blockPct > 20) { status = 'Bottleneck'; statusColor = '#e74c3c'; }
        else if (starvePct > 30) { status = 'Underutilized'; statusColor = '#f39c12'; }
        else if (utilization > 100) { status = 'Overloaded'; statusColor = '#e74c3c'; }
        else if (downPct > 10) { status = 'Unreliable'; statusColor = '#666'; }

        html += `
            <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 8px;">${station.name}</td>
                <td style="padding: 8px; text-align: right;">${station.cycleTime}s</td>
                <td style="padding: 8px; text-align: right;">${state.processed}</td>
                <td style="padding: 8px; text-align: right; color: #4a9f6e;">${workPct.toFixed(1)}%</td>
                <td style="padding: 8px; text-align: right; color: #e74c3c;">${blockPct.toFixed(1)}%</td>
                <td style="padding: 8px; text-align: right; color: #f39c12;">${starvePct.toFixed(1)}%</td>
                ${breakdownsEnabled ? `<td style="padding: 8px; text-align: right; color: #666;">${downPct.toFixed(1)}%</td>` : ''}
                <td style="padding: 8px; color: ${statusColor};">${status}</td>
            </tr>
        `;
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

function updateLineBottleneck(takt) {
    const container = document.getElementById('ls-bottleneck');

    // Find the station with highest blocked time
    let bottleneckIdx = -1;
    let maxBlockTime = 0;
    lineSimState.stationStats.forEach((state, i) => {
        if (state.blocked > maxBlockTime) {
            maxBlockTime = state.blocked;
            bottleneckIdx = i;
        }
    });

    // Also check for theoretical bottleneck (highest CT)
    let theoreticalBottleneck = 0;
    let maxCT = 0;
    lineStations.forEach((s, i) => {
        if (s.cycleTime > maxCT) {
            maxCT = s.cycleTime;
            theoreticalBottleneck = i;
        }
    });

    const theoreticalThroughput = 3600 / maxCT;
    const actualThroughput = lineSimState.time > 0 ? (lineSimState.completed / lineSimState.time) * 3600 : 0;
    const lostThroughput = theoreticalThroughput - actualThroughput;

    // Check flow mode
    const flowMode = document.getElementById('ls-flow-mode').value;
    const batchSize = flowMode === 'batch' ? parseInt(document.getElementById('ls-batch-size').value) || 5 : 1;
    const breakdownsEnabled = document.getElementById('ls-breakdowns').checked;

    // Calculate total WIP and downtime
    const totalWip = lineSimState.stationStats.reduce((sum, s) => sum + s.outputBuffer + (s.hasUnit ? (s.batchCount || 1) : 0), 0);
    const totalDowntime = lineSimState.stationStats.reduce((sum, s) => sum + (s.downtime || 0), 0);
    const downtimePct = lineSimState.time > 0 ? (totalDowntime / (lineSimState.time * lineStations.length)) * 100 : 0;

    let html = '';

    if (lineSimState.time < 30) {
        html = '<em>Running simulation... gathering data...</em>';
    } else {
        let improvementHtml = '';

        // Takt violation
        if (maxCT > takt) {
            improvementHtml += `<div style="color: #e74c3c; margin-bottom: 4px;">⚠ ${lineStations[theoreticalBottleneck].name} CT (${maxCT}s) exceeds takt (${takt}s)</div>`;
        }

        // Variability cost
        if (lostThroughput > 5) {
            improvementHtml += `<div style="margin-bottom: 4px;"><svg style="display:inline-block;vertical-align:middle;width:14px;height:14px;margin-right:4px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>Variability costing ~${lostThroughput.toFixed(1)} units/hr</div>`;
        }

        // Batch mode impact
        if (flowMode === 'batch') {
            const wipWithOnePiece = lineStations.length; // theoretical minimum
            const wipMultiplier = totalWip / wipWithOnePiece;
            improvementHtml += `<div style="color: #f39c12; margin-bottom: 4px;"><svg style="display:inline-block;vertical-align:middle;width:14px;height:14px;margin-right:4px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8l-9-5-9 5v8l9 5z"/><path d="M3.3 7L12 12l8.7-5"/><line x1="12" y1="22" x2="12" y2="12"/></svg>Batch mode: WIP is ~${wipMultiplier.toFixed(1)}x higher than one-piece flow</div>`;
            improvementHtml += `<div style="margin-bottom: 4px;"><span style="opacity:0.7;">Tip:</span> Switch to one-piece flow to reduce WIP from ${totalWip} to ~${wipWithOnePiece}</div>`;
        }

        // Buffer suggestion
        if (bottleneckIdx >= 0 && bottleneckIdx !== theoreticalBottleneck && maxBlockTime > 10) {
            improvementHtml += `<div style="margin-bottom: 4px;"><svg style="display:inline-block;vertical-align:middle;width:14px;height:14px;margin-right:4px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.5 9a9 9 0 0 1 14.8-3.4L23 10M1 14l4.6 4.4A9 9 0 0 0 20.5 15"/></svg>Buffer before ${lineStations[bottleneckIdx].name} may need increase</div>`;
        }

        // Downtime impact
        if (breakdownsEnabled && downtimePct > 5) {
            const throughputLostToDowntime = theoreticalThroughput * (downtimePct / 100);
            improvementHtml += `<div style="color: #666; margin-bottom: 4px;"><svg style="display:inline-block;vertical-align:middle;width:14px;height:14px;margin-right:4px;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>Downtime (${downtimePct.toFixed(1)}%) costing ~${throughputLostToDowntime.toFixed(1)} units/hr</div>`;
            improvementHtml += `<div style="margin-bottom: 4px;"><span style="opacity:0.7;">Tip:</span> TPM focus: improve MTBF or reduce repair time</div>`;
        }

        // Default suggestion
        if (!improvementHtml) {
            improvementHtml = `<div style="margin-bottom: 4px;"><span style="opacity:0.7;">Tip:</span> Reduce ${lineStations[theoreticalBottleneck].name} CT or add parallel capacity</div>`;
        }

        html = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div>
                    <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">Constraint Analysis</div>
                    <div style="margin-bottom: 4px;"><strong>Theoretical bottleneck:</strong> ${lineStations[theoreticalBottleneck].name} (CT: ${maxCT}s)</div>
                    <div style="margin-bottom: 4px;"><strong>Max theoretical throughput:</strong> ${theoreticalThroughput.toFixed(1)} units/hr</div>
                    <div style="margin-bottom: 4px;"><strong>Actual throughput:</strong> ${actualThroughput.toFixed(1)} units/hr (${((actualThroughput/theoreticalThroughput)*100).toFixed(0)}%)</div>
                    ${bottleneckIdx >= 0 && maxBlockTime > 10 ? `
                        <div style="margin-bottom: 4px;"><strong>Actual bottleneck:</strong> <span style="color: #e74c3c;">${lineStations[bottleneckIdx].name}</span> (${(maxBlockTime).toFixed(0)}s blocked)</div>
                    ` : ''}
                    <div style="margin-bottom: 4px;"><strong>Total WIP:</strong> ${totalWip} units</div>
                </div>
                <div>
                    <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">Improvement Opportunities</div>
                    ${improvementHtml}
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

function resetLineSim() {
    lineSimState.running = false;
    if (lineSimState.interval) clearInterval(lineSimState.interval);
    lineSimState.time = 0;
    lineSimState.completed = 0;
    lineSimState.history = { time: [], wip: [], throughput: [] };
    lineSimState.stationStats = [];
    lineSimState.currentOrder = null;
    lineSimState.currentOrderProgress = 0;
    lineSimState.currentProduct = null;
    lineSimState.changingOver = false;
    lineSimState.changeoverRemaining = 0;
    lineSimState.totalChangeoverTime = 0;
    lineSimState.orderResults = [];

    document.getElementById('ls-start').innerHTML = '▶ Start Line';
    document.getElementById('ls-completed').textContent = '0';
    document.getElementById('ls-throughput').innerHTML = '0<span class="calc-result-unit">/hr</span>';
    document.getElementById('ls-total-wip').textContent = '0';
    document.getElementById('ls-efficiency').innerHTML = '0<span class="calc-result-unit">%</span>';
    document.getElementById('ls-station-stats').innerHTML = '';
    document.getElementById('ls-bottleneck').innerHTML = '<em>Start simulation to identify bottleneck...</em>';

    // Reset order statuses
    lineOrders.forEach(o => { o.status = 'pending'; o.completedTime = null; });
    renderOrders();
    updateDeliveryMetrics();

    document.getElementById('ls-chart').innerHTML = '';
    renderLineVisual();
}

// ============================================================================
// Order-Driven Mode Functions
// ============================================================================

function updateSimMode() {
    const mode = document.getElementById('ls-sim-mode').value;
    const orderPanel = document.getElementById('ls-order-panel');
    const deliverySection = document.getElementById('ls-delivery-section');
    const hint = document.getElementById('ls-sim-mode-hint');

    if (mode === 'orders') {
        orderPanel.style.display = 'block';
        deliverySection.style.display = 'block';
        hint.textContent = 'Process customer orders with due dates';
        hint.style.color = 'var(--accent)';
        renderProducts();
        renderOrders();
    } else {
        orderPanel.style.display = 'none';
        deliverySection.style.display = 'none';
        hint.textContent = 'Continuous production vs customer orders';
        hint.style.color = '';
    }
    resetLineSim();
}

// Product management
function renderProducts() {
    const container = document.getElementById('ls-products');
    container.innerHTML = lineProducts.map((p, i) => `
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px; background: var(--bg-card); border-radius: 6px; border-left: 4px solid ${p.color};">
            <input type="text" value="${p.name}" onchange="lineProducts[${i}].name = this.value; renderOrders();"
                   style="flex: 1; padding: 4px 8px; background: transparent; border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary); font-size: 12px;">
            <span style="font-size: 11px; color: var(--text-dim);">CT×</span>
            <input type="number" value="${p.ctMultiplier}" step="0.1" min="0.5" max="3"
                   onchange="lineProducts[${i}].ctMultiplier = parseFloat(this.value) || 1;"
                   style="width: 50px; padding: 4px; background: transparent; border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary); font-size: 12px; text-align: center;">
            <input type="color" value="${p.color}" onchange="lineProducts[${i}].color = this.value; renderProducts(); renderOrders();"
                   style="width: 24px; height: 24px; border: none; cursor: pointer;">
            ${lineProducts.length > 1 ? `<button onclick="removeProduct(${i})" style="background: none; border: none; color: var(--text-dim); cursor: pointer;">&times;</button>` : ''}
        </div>
    `).join('');
}

function addProduct() {
    const colors = ['#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e91e63'];
    const nextColor = colors[lineProducts.length % colors.length];
    const nextId = String.fromCharCode(65 + lineProducts.length); // A, B, C...
    lineProducts.push({
        id: nextId,
        name: `Product ${nextId}`,
        color: nextColor,
        ctMultiplier: 1.0 + (Math.random() * 0.4 - 0.2) // 0.8 to 1.2
    });
    renderProducts();
}

function removeProduct(idx) {
    if (lineProducts.length > 1) {
        lineProducts.splice(idx, 1);
        renderProducts();
        // Remove orders with this product
        lineOrders = lineOrders.filter(o => o.productIdx < lineProducts.length);
        renderOrders();
    }
}

// Order management
function renderOrders() {
    const tbody = document.getElementById('ls-orders-body');
    if (!tbody) return;

    if (lineOrders.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="padding: 12px; text-align: center; color: var(--text-dim);"><em>No orders — click "Generate Sample" or add manually</em></td></tr>';
        return;
    }

    tbody.innerHTML = lineOrders.map((o, i) => {
        const product = lineProducts[o.productIdx] || lineProducts[0];
        const statusColor = o.status === 'complete' ? (o.wasLate ? '#e74c3c' : '#4a9f6e') :
                           o.status === 'in-progress' ? '#f39c12' : 'var(--text-dim)';
        const statusText = o.status === 'complete' ? (o.wasLate ? 'LATE' : 'ON TIME') :
                          o.status === 'in-progress' ? 'IN PROGRESS' : 'Pending';

        return `
            <tr style="border-bottom: 1px solid var(--border);">
                <td style="padding: 6px; font-weight: 600;">#${o.id}</td>
                <td style="padding: 6px;"><span style="display: inline-block; width: 8px; height: 8px; background: ${product.color}; border-radius: 2px; margin-right: 6px;"></span>${product.name}</td>
                <td style="padding: 6px; text-align: right;">${o.qty}</td>
                <td style="padding: 6px; text-align: right;">${o.dueTime}s</td>
                <td style="padding: 6px; text-align: center; color: ${statusColor}; font-weight: 600; font-size: 11px;">${statusText}</td>
                <td style="padding: 6px;"><button onclick="removeOrder(${i})" style="background: none; border: none; color: var(--text-dim); cursor: pointer; font-size: 14px;">&times;</button></td>
            </tr>
        `;
    }).join('');
}

function addOrder() {
    lineOrders.push({
        id: orderIdCounter++,
        productIdx: lineOrders.length % lineProducts.length,
        qty: 5 + Math.floor(Math.random() * 10),
        dueTime: 300 + lineOrders.length * 200 + Math.floor(Math.random() * 100),
        status: 'pending',
        completedTime: null,
        wasLate: false,
        lateReason: null,
    });
    renderOrders();
}

function removeOrder(idx) {
    lineOrders.splice(idx, 1);
    renderOrders();
}

function generateSampleOrders() {
    lineOrders = [];
    orderIdCounter = 1;

    // Generate 6-8 orders with realistic timing
    const numOrders = 6 + Math.floor(Math.random() * 3);
    let cumulativeTime = 200;

    for (let i = 0; i < numOrders; i++) {
        const productIdx = i % lineProducts.length;
        const qty = 3 + Math.floor(Math.random() * 8);
        // Due time gives some slack but not too much
        const baseLeadTime = qty * 60; // ~60s per unit estimate
        const slack = 50 + Math.random() * 100;

        lineOrders.push({
            id: orderIdCounter++,
            productIdx,
            qty,
            dueTime: Math.round(cumulativeTime + baseLeadTime + slack),
            status: 'pending',
            completedTime: null,
            wasLate: false,
            lateReason: null,
        });

        cumulativeTime += baseLeadTime * 0.8; // Orders overlap slightly
    }

    renderOrders();
}

// Helper to update metrics during changeover or waiting
function updateOrderMetrics(simSecondsPerTick) {
    const takt = parseFloat(document.getElementById('ls-takt').value) || 60;
    const maxWip = parseInt(document.getElementById('ls-max-wip').value) || 5;

    // Update regular metrics even during changeover
    const totalWip = lineSimState.stationStats.reduce((sum, s) => sum + s.outputBuffer + (s.hasUnit ? (s.batchCount || 1) : 0), 0);
    const throughputPerHour = lineSimState.time > 0 ? (lineSimState.completed / lineSimState.time) * 3600 : 0;

    document.getElementById('ls-completed').textContent = lineSimState.completed;
    document.getElementById('ls-throughput').innerHTML = `${throughputPerHour.toFixed(1)}<span class="calc-result-unit">/hr</span>`;
    document.getElementById('ls-total-wip').textContent = totalWip;

    // Update delivery metrics
    updateDeliveryMetrics();

    // Update visual
    updateLineVisual(maxWip);
}

function updateDeliveryMetrics() {
    const completed = lineSimState.orderResults.filter(r => r.completedTime !== null);
    const onTime = completed.filter(r => !r.wasLate);
    const late = completed.filter(r => r.wasLate);

    const totalOrders = lineOrders.length;
    const otdPct = completed.length > 0 ? (onTime.length / completed.length) * 100 : 0;
    const avgLead = completed.length > 0 ?
        completed.reduce((sum, r) => sum + r.completedTime, 0) / completed.length : 0;

    document.getElementById('ls-otd').innerHTML = completed.length > 0 ?
        `<span style="color: ${otdPct >= 95 ? '#4a9f6e' : otdPct >= 80 ? '#f39c12' : '#e74c3c'}">${otdPct.toFixed(0)}</span><span class="calc-result-unit">%</span>` :
        '—<span class="calc-result-unit">%</span>';
    document.getElementById('ls-orders-complete').innerHTML = `${completed.length}<span class="calc-result-unit">/ ${totalOrders}</span>`;
    document.getElementById('ls-avg-lead').innerHTML = completed.length > 0 ?
        `${avgLead.toFixed(0)}<span class="calc-result-unit">sec</span>` : '—<span class="calc-result-unit">sec</span>';
    document.getElementById('ls-changeover-loss').innerHTML = `${lineSimState.totalChangeoverTime.toFixed(0)}<span class="calc-result-unit">sec</span>`;

    // Late order analysis
    const lateAnalysis = document.getElementById('ls-late-analysis');
    const lateOrdersDiv = document.getElementById('ls-late-orders');

    if (late.length > 0) {
        lateAnalysis.style.display = 'block';
        lateOrdersDiv.innerHTML = late.map(r => {
            const product = lineProducts[r.productIdx] || lineProducts[0];
            return `
                <div style="margin-bottom: 8px; padding: 8px; background: rgba(231, 76, 60, 0.1); border-radius: 4px;">
                    <div><strong>Order #${r.orderId}</strong> (${product.name}, ${r.qty} units)</div>
                    <div style="font-size: 12px;">Due: ${r.dueTime}s | Completed: ${r.completedTime.toFixed(0)}s | <span style="color: #e74c3c;">Late by ${(r.completedTime - r.dueTime).toFixed(0)}s</span></div>
                    ${r.lateReason ? `<div style="font-size: 12px; color: #e74c3c; margin-top: 4px;">&#9656; ${r.lateReason}</div>` : ''}
                </div>
            `;
        }).join('');
    } else {
        lateAnalysis.style.display = 'none';
    }
}

// Import stations from Yamazumi chart
function importFromYamazumi() {
    if (yamazumiData && yamazumiData.length > 0) {
        lineStations = yamazumiData.map(s => ({
            name: s.name,
            cycleTime: s.time
        }));
        renderLineStations();
        resetLineSim();
    } else {
        alert('No Yamazumi data found. Add stations to Yamazumi first.');
    }
}

// Toggle A/B compare panel
function toggleLineCompare() {
    const panel = document.getElementById('ls-compare-panel');
    const btn = document.getElementById('ls-compare-btn');
    if (panel.style.display === 'none') {
        panel.style.display = 'block';
        btn.style.background = 'var(--accent)';
        btn.style.color = 'white';
        btn.style.borderColor = 'var(--accent)';
        renderCompareStationOverrides();
    } else {
        panel.style.display = 'none';
        btn.style.background = 'var(--bg-secondary)';
        btn.style.color = 'var(--text-secondary)';
        btn.style.borderColor = 'var(--border)';
    }
}

function renderCompareStationOverrides() {
    const container = document.getElementById('ls-compare-b-stations');
    if (!container) return;

    container.innerHTML = lineStations.map((s, i) => `
        <div style="display: flex; align-items: center; gap: 4px; padding: 4px 8px; background: var(--bg-card); border-radius: 4px;">
            <span style="font-size: 11px; color: var(--text-dim); width: 60px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${s.name}</span>
            <input type="number" id="ls-compare-b-st-${i}" value="" placeholder="${s.cycleTime}"
                   style="width: 50px; padding: 2px 4px; background: transparent; border: 1px solid var(--border); border-radius: 3px; color: var(--text-primary); font-size: 11px; text-align: center;">
        </div>
    `).join('');
}

function applyComparePreset(preset) {
    const cov = parseFloat(document.getElementById('ls-cov').value);
    const maxWip = parseInt(document.getElementById('ls-max-wip').value) || 5;
    const batchSize = parseInt(document.getElementById('ls-batch-size').value) || 1;

    // Find bottleneck
    let bottleneckIdx = 0;
    let maxCT = 0;
    lineStations.forEach((s, i) => {
        if (s.cycleTime > maxCT) { maxCT = s.cycleTime; bottleneckIdx = i; }
    });

    // Clear all first
    resetCompareB();

    switch (preset) {
        case 'reduce-bottleneck':
            document.getElementById(`ls-compare-b-st-${bottleneckIdx}`).value = Math.round(maxCT * 0.9);
            break;
        case 'add-buffer':
            document.getElementById('ls-compare-b-buffer').value = maxWip * 2;
            break;
        case 'one-piece':
            document.getElementById('ls-compare-b-batch').value = 1;
            break;
        case 'reduce-cov':
            document.getElementById('ls-compare-b-cov').value = Math.round(cov / 2);
            break;
    }
}

function resetCompareB() {
    document.getElementById('ls-compare-b-buffer').value = '';
    document.getElementById('ls-compare-b-cov').value = '';
    document.getElementById('ls-compare-b-batch').value = '';
    document.getElementById('ls-compare-b-ct-adj').value = '';
    lineStations.forEach((s, i) => {
        const input = document.getElementById(`ls-compare-b-st-${i}`);
        if (input) input.value = '';
    });
}

// Run A/B comparison simulation
function runLineCompare() {
    const cov = parseFloat(document.getElementById('ls-cov').value) / 100;
    const maxWip = parseInt(document.getElementById('ls-max-wip').value) || 5;
    const flowMode = document.getElementById('ls-flow-mode').value;
    const batchSize = flowMode === 'batch' ? parseInt(document.getElementById('ls-batch-size').value) || 5 : 1;

    // Get simulation settings
    const simDuration = parseInt(document.getElementById('ls-compare-duration').value) || 300;
    const warmupTime = parseInt(document.getElementById('ls-compare-warmup').value) || 0;

    // Generate random seed for fair comparison
    const seed = Math.random();

    // Status update
    const statusEl = document.getElementById('ls-compare-status');
    statusEl.textContent = 'Running simulations...';

    // Use setTimeout to allow UI update
    setTimeout(() => {
        // Run scenario A (current state)
        const resultA = runLineSimWithWarmup(lineStations, cov, maxWip, batchSize, seed, simDuration, warmupTime);

        // Build scenario B from inputs
        let stationsB = JSON.parse(JSON.stringify(lineStations));
        let covB = cov;
        let maxWipB = maxWip;
        let batchSizeB = batchSize;

        // Apply custom buffer
        const bufferB = document.getElementById('ls-compare-b-buffer').value;
        if (bufferB) maxWipB = parseInt(bufferB);

        // Apply custom CoV
        const covBInput = document.getElementById('ls-compare-b-cov').value;
        if (covBInput) covB = parseFloat(covBInput) / 100;

        // Apply custom batch size
        const batchB = document.getElementById('ls-compare-b-batch').value;
        if (batchB) batchSizeB = parseInt(batchB);

        // Apply bottleneck CT adjustment (percentage)
        const ctAdj = document.getElementById('ls-compare-b-ct-adj').value;
        if (ctAdj) {
            let bottleneckIdx = 0;
            let maxCT = 0;
            stationsB.forEach((s, i) => {
                if (s.cycleTime > maxCT) { maxCT = s.cycleTime; bottleneckIdx = i; }
            });
            stationsB[bottleneckIdx].cycleTime *= (1 + parseFloat(ctAdj) / 100);
        }

        // Apply station-specific overrides
        lineStations.forEach((s, i) => {
            const override = document.getElementById(`ls-compare-b-st-${i}`)?.value;
            if (override) stationsB[i].cycleTime = parseFloat(override);
        });

        const resultB = runLineSimWithWarmup(stationsB, covB, maxWipB, batchSizeB, seed, simDuration, warmupTime);

        // Display results
        const warmupNote = warmupTime > 0 ? ` <span style="color: var(--text-dim);">(after ${warmupTime}s warmup)</span>` : '';

        document.getElementById('ls-compare-a-stats').innerHTML = `
            <div style="margin-bottom: 4px;"><strong>Throughput:</strong> ${resultA.throughput.toFixed(1)}/hr</div>
            <div style="margin-bottom: 4px;"><strong>Units:</strong> ${resultA.completed}</div>
            <div style="margin-bottom: 4px;"><strong>Avg WIP:</strong> ${resultA.avgWip.toFixed(1)}</div>
            <div><strong>Efficiency:</strong> ${resultA.efficiency.toFixed(1)}%</div>
        `;

        document.getElementById('ls-compare-b-stats').innerHTML = `
            <div style="margin-bottom: 4px;"><strong>Throughput:</strong> ${resultB.throughput.toFixed(1)}/hr</div>
            <div style="margin-bottom: 4px;"><strong>Units:</strong> ${resultB.completed}</div>
            <div style="margin-bottom: 4px;"><strong>Avg WIP:</strong> ${resultB.avgWip.toFixed(1)}</div>
            <div><strong>Efficiency:</strong> ${resultB.efficiency.toFixed(1)}%</div>
        `;

        // Verdict
        const throughputDelta = resultA.throughput > 0 ? ((resultB.throughput - resultA.throughput) / resultA.throughput) * 100 : 0;
        const wipDelta = resultA.avgWip > 0 ? ((resultB.avgWip - resultA.avgWip) / resultA.avgWip) * 100 : 0;

        let verdict = '';
        if (throughputDelta > 5) {
            verdict = `<span style="color: #4a9f6e; font-weight: 600;">✓ B wins: +${throughputDelta.toFixed(1)}% throughput</span>`;
        } else if (throughputDelta < -5) {
            verdict = `<span style="color: #e74c3c; font-weight: 600;">✗ A wins: B is ${Math.abs(throughputDelta).toFixed(1)}% worse</span>`;
        } else {
            verdict = `<span style="color: #f39c12;">≈ Negligible difference (${throughputDelta > 0 ? '+' : ''}${throughputDelta.toFixed(1)}%)</span>`;
        }

        if (Math.abs(wipDelta) > 10) {
            verdict += `<span style="margin-left: 16px; color: var(--text-dim);">WIP: ${wipDelta > 0 ? '+' : ''}${wipDelta.toFixed(0)}%</span>`;
        }

        document.getElementById('ls-compare-verdict').innerHTML = verdict;
        statusEl.textContent = `Simulated ${simDuration}s${warmupTime > 0 ? ` (+ ${warmupTime}s warmup)` : ''}`;
    }, 10);
}

// Run simulation with optional warmup (non-visual, for comparison)
function runLineSimWithWarmup(stations, cov, maxWip, batchSize, seed, duration = 300, warmup = 0) {
    // Seeded random for reproducibility
    let rng = seed;
    const seededRandom = () => {
        rng = (rng * 9301 + 49297) % 233280;
        return rng / 233280;
    };

    const simDuration = duration + warmup;
    const dt = 0.5;

    let state = stations.map(() => ({
        working: 0, blocked: 0, starved: 0,
        timeRemaining: 0, outputBuffer: 0, hasUnit: false, batchCount: 0
    }));

    let completed = 0;
    let completedDuringMeasure = 0;
    let totalWip = 0;
    let wipSamples = 0;
    let inMeasurePhase = warmup === 0;

    for (let t = 0; t < simDuration; t += dt) {
        // Check if we just entered measure phase
        if (!inMeasurePhase && t >= warmup) {
            inMeasurePhase = true;
            completedDuringMeasure = 0; // Reset for measurement
        }
        // Process stations (last to first)
        for (let i = stations.length - 1; i >= 0; i--) {
            const station = stations[i];
            const st = state[i];
            const prevState = i > 0 ? state[i - 1] : null;
            const isFirst = i === 0;
            const isLast = i === stations.length - 1;

            const actualCT = station.cycleTime * (1 + (seededRandom() - 0.5) * 2 * cov);

            if (st.hasUnit) {
                st.timeRemaining -= dt;
                if (st.timeRemaining <= 0) {
                    if (isLast) {
                        completed += batchSize;
                        if (inMeasurePhase) completedDuringMeasure += batchSize;
                        st.hasUnit = false;
                        st.batchCount = 0;
                    } else if (st.outputBuffer < maxWip) {
                        st.outputBuffer += batchSize;
                        st.hasUnit = false;
                        st.batchCount = 0;
                    } else {
                        st.blocked += dt;
                    }
                } else {
                    st.working += dt;
                }
            } else {
                if (isFirst) {
                    st.hasUnit = true;
                    st.batchCount = batchSize;
                    st.timeRemaining = actualCT * batchSize;
                } else if (prevState && prevState.outputBuffer >= batchSize) {
                    prevState.outputBuffer -= batchSize;
                    st.hasUnit = true;
                    st.batchCount = batchSize;
                    st.timeRemaining = actualCT * batchSize;
                } else {
                    st.starved += dt;
                }
            }
        }

        // Only count metrics after warmup
        if (t >= warmup) {
            const wip = state.reduce((sum, s) => sum + s.outputBuffer + (s.hasUnit ? s.batchCount : 0), 0);
            totalWip += wip;
            wipSamples++;
        }
    }

    const totalWorking = state.reduce((sum, s) => sum + s.working, 0);
    const totalTime = state.reduce((sum, s) => sum + s.working + s.blocked + s.starved, 0);
    const measureDuration = duration; // Only the non-warmup portion

    return {
        completed: completedDuringMeasure,
        throughput: measureDuration > 0 ? (completedDuringMeasure / measureDuration) * 3600 : 0,
        avgWip: wipSamples > 0 ? totalWip / wipSamples : 0,
        efficiency: totalTime > 0 ? (totalWorking / totalTime) * 100 : 0,
    };
}

// Scenario management (localStorage)
const LINE_SCENARIOS_KEY = 'svend_line_scenarios';

function getLineScenarios() {
    try {
        return JSON.parse(localStorage.getItem(LINE_SCENARIOS_KEY) || '{}');
    } catch {
        return {};
    }
}

function saveLineScenario() {
    const name = document.getElementById('ls-scenario-name').value.trim();
    if (!name) {
        alert('Please enter a scenario name');
        return;
    }

    const scenario = {
        stations: JSON.parse(JSON.stringify(lineStations)),
        takt: document.getElementById('ls-takt').value,
        cov: document.getElementById('ls-cov').value,
        maxWip: document.getElementById('ls-max-wip').value,
        flowMode: document.getElementById('ls-flow-mode').value,
        batchSize: document.getElementById('ls-batch-size').value,
        breakdowns: document.getElementById('ls-breakdowns').checked,
        savedAt: new Date().toISOString(),
    };

    const scenarios = getLineScenarios();
    scenarios[name] = scenario;
    localStorage.setItem(LINE_SCENARIOS_KEY, JSON.stringify(scenarios));

    document.getElementById('ls-scenario-name').value = '';
    refreshScenarioDropdown();
    alert(`Scenario "${name}" saved!`);
}

function loadLineScenario() {
    const select = document.getElementById('ls-scenarios');
    const name = select.value;
    if (!name) return;

    const scenarios = getLineScenarios();
    const scenario = scenarios[name];
    if (!scenario) return;

    // Stop any running simulation
    resetLineSim();

    // Load settings
    lineStations = JSON.parse(JSON.stringify(scenario.stations));
    document.getElementById('ls-takt').value = scenario.takt || 60;
    document.getElementById('ls-cov').value = scenario.cov || 15;
    document.getElementById('ls-max-wip').value = scenario.maxWip || 5;
    document.getElementById('ls-flow-mode').value = scenario.flowMode || 'one-piece';
    document.getElementById('ls-batch-size').value = scenario.batchSize || 5;
    document.getElementById('ls-breakdowns').checked = scenario.breakdowns || false;

    renderLineStations();
    updateLineSimParams();
}

function deleteLineScenario() {
    const select = document.getElementById('ls-scenarios');
    const name = select.value;
    if (!name) return;

    if (!confirm(`Delete scenario "${name}"?`)) return;

    const scenarios = getLineScenarios();
    delete scenarios[name];
    localStorage.setItem(LINE_SCENARIOS_KEY, JSON.stringify(scenarios));
    refreshScenarioDropdown();
}

function refreshScenarioDropdown() {
    const select = document.getElementById('ls-scenarios');
    const scenarios = getLineScenarios();

    select.innerHTML = '<option value="">— Select saved scenario —</option>';
    Object.keys(scenarios).sort().forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
    });
}

function exportLineReport() {
    if (lineSimState.time < 10) {
        alert('Run the simulation first to generate data for the report.');
        return;
    }

    const takt = parseFloat(document.getElementById('ls-takt').value) || 60;
    const cov = parseFloat(document.getElementById('ls-cov').value);
    const maxWip = parseInt(document.getElementById('ls-max-wip').value) || 5;
    const flowMode = document.getElementById('ls-flow-mode').value;
    const batchSize = flowMode === 'batch' ? parseInt(document.getElementById('ls-batch-size').value) || 5 : 1;

    // Find bottleneck
    let bottleneckIdx = 0;
    let maxCT = 0;
    lineStations.forEach((s, i) => {
        if (s.cycleTime > maxCT) { maxCT = s.cycleTime; bottleneckIdx = i; }
    });

    const theoreticalThroughput = 3600 / maxCT;
    const actualThroughput = (lineSimState.completed / lineSimState.time) * 3600;
    const totalWip = lineSimState.stationStats.reduce((sum, s) => sum + s.outputBuffer + (s.hasUnit ? (s.batchCount || 1) : 0), 0);

    let report = `LINE SIMULATION REPORT
Generated: ${new Date().toLocaleString()}
================================================================================

CONFIGURATION
─────────────────────────────────────────────────────────────────────────────────
Takt Time:           ${takt} seconds
Variability (CoV):   ${cov}%
Buffer Capacity:     ${maxWip} units per station
Flow Mode:           ${flowMode === 'batch' ? `Batch (${batchSize} units)` : 'One-Piece Flow'}

STATIONS
─────────────────────────────────────────────────────────────────────────────────
`;

    lineStations.forEach((station, i) => {
        const state = lineSimState.stationStats[i];
        const totalTime = state.working + state.blocked + state.starved + (state.downtime || 0);
        const workPct = totalTime > 0 ? (state.working / totalTime) * 100 : 0;
        const blockPct = totalTime > 0 ? (state.blocked / totalTime) * 100 : 0;
        const starvePct = totalTime > 0 ? (state.starved / totalTime) * 100 : 0;

        report += `${station.name.padEnd(15)} CT: ${station.cycleTime}s | Processed: ${state.processed} | Working: ${workPct.toFixed(1)}% | Blocked: ${blockPct.toFixed(1)}% | Starved: ${starvePct.toFixed(1)}%\n`;
    });

    report += `
RESULTS (${(lineSimState.time / 60).toFixed(1)} minutes simulated)
─────────────────────────────────────────────────────────────────────────────────
Units Completed:       ${lineSimState.completed}
Actual Throughput:     ${actualThroughput.toFixed(1)} units/hour
Theoretical Max:       ${theoreticalThroughput.toFixed(1)} units/hour
Efficiency:            ${((actualThroughput / theoreticalThroughput) * 100).toFixed(1)}%
Total WIP:             ${totalWip} units

CONSTRAINT ANALYSIS
─────────────────────────────────────────────────────────────────────────────────
Bottleneck Station:    ${lineStations[bottleneckIdx].name} (CT: ${maxCT}s)
Capacity Loss:         ${(theoreticalThroughput - actualThroughput).toFixed(1)} units/hour
`;

    if (flowMode === 'batch') {
        report += `\n⚠ BATCH MODE ACTIVE: WIP is elevated due to batch processing. Consider one-piece flow.\n`;
    }

    if (maxCT > takt) {
        report += `\n⚠ TAKT VIOLATION: ${lineStations[bottleneckIdx].name} CT (${maxCT}s) exceeds takt time (${takt}s).\n`;
    }

    report += `
================================================================================
Report generated by SVEND Operations Workbench
`;

    // Copy to clipboard and offer download
    navigator.clipboard.writeText(report).then(() => {
        const blob = new Blob([report], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `line-sim-report-${new Date().toISOString().slice(0, 10)}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    });
}

function initLineSim() {
    renderLineStations();
    updateLineSimParams();
    renderCompareStationOverrides();
    refreshScenarioDropdown();
    renderProducts();
    generateSampleOrders(); // Start with sample orders
}
