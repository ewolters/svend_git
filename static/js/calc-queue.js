/**
 * calc-queue.js — Queue Theory Calculators
 *
 * Load order: after calc-core.js (uses SvendOps, renderNextSteps, MonteCarlo)
 * Extracted from: calculators.html (inline script)
 *
 * Provides: M/M/c queue, M/M/c/K finite queue, priority queue,
 * staffing optimizer, multi-stage tandem queue analysis.
 */

// ============================================================================
// Queuing Theory (M/M/c)
// ============================================================================

function factorial(n) {
    if (n <= 1) return 1;
    let result = 1;
    for (let i = 2; i <= n; i++) result *= i;
    return result;
}

function calcQueue() {
    const lambda = parseFloat(document.getElementById('queue-lambda').value) || 0;
    const mu = parseFloat(document.getElementById('queue-mu').value) || 1;
    const c = parseInt(document.getElementById('queue-c').value) || 1;

    const rho = lambda / (c * mu);
    const r = lambda / mu;

    // Check stability
    if (rho >= 1) {
        document.getElementById('queue-rho').innerHTML = `${(rho * 100).toFixed(1)}<span class="calc-result-unit">%</span>`;
        document.getElementById('queue-stable').innerHTML = '<span style="color:#e74c3c">No (λ ≥ cμ) - System unstable!</span>';
        document.getElementById('queue-wq').innerHTML = '∞';
        document.getElementById('queue-lq').textContent = '∞';
        document.getElementById('queue-w').innerHTML = '∞';
        document.getElementById('queue-l').textContent = '∞';
        document.getElementById('queue-pw').innerHTML = '100<span class="calc-result-unit">%</span>';
        document.getElementById('queue-capacity').textContent = `${(c * mu).toFixed(1)} customers/hr`;
        return;
    }

    // P0 calculation (probability of empty system)
    let sum = 0;
    for (let n = 0; n < c; n++) {
        sum += Math.pow(r, n) / factorial(n);
    }
    sum += (Math.pow(r, c) / factorial(c)) * (1 / (1 - rho));
    const P0 = 1 / sum;

    // Probability of waiting (Erlang C formula)
    const Pc = (Math.pow(r, c) / factorial(c)) * (1 / (1 - rho)) * P0;

    // Queue length
    const Lq = Pc * rho / (1 - rho);

    // Wait time in queue
    const Wq = Lq / lambda;

    // Total time in system
    const W = Wq + (1 / mu);

    // Total in system
    const L = lambda * W;

    document.getElementById('queue-rho').innerHTML = `${(rho * 100).toFixed(1)}<span class="calc-result-unit">%</span>`;
    document.getElementById('queue-wq').innerHTML = `${(Wq * 60).toFixed(1)}<span class="calc-result-unit">min</span>`;
    document.getElementById('queue-lq').textContent = Lq.toFixed(2);
    document.getElementById('queue-w').innerHTML = `${(W * 60).toFixed(1)}<span class="calc-result-unit">min</span>`;
    document.getElementById('queue-l').textContent = L.toFixed(2);
    document.getElementById('queue-pw').innerHTML = `${(Pc * 100).toFixed(1)}<span class="calc-result-unit">%</span>`;
    document.getElementById('queue-stable').innerHTML = '<span style="color:#27ae60">Yes (λ < cμ)</span>';
    document.getElementById('queue-capacity').textContent = `${(c * mu).toFixed(1)} customers/hr`;

    // Wait time vs utilization chart
    const utils = [];
    const waitTimes = [];
    for (let u = 0.1; u < 0.99; u += 0.02) {
        const testLambda = u * c * mu;
        const testR = testLambda / mu;
        let testSum = 0;
        for (let n = 0; n < c; n++) {
            testSum += Math.pow(testR, n) / factorial(n);
        }
        testSum += (Math.pow(testR, c) / factorial(c)) * (1 / (1 - u));
        const testP0 = 1 / testSum;
        const testPc = (Math.pow(testR, c) / factorial(c)) * (1 / (1 - u)) * testP0;
        const testLq = testPc * u / (1 - u);
        const testWq = testLq / testLambda * 60; // in minutes

        utils.push(u * 100);
        waitTimes.push(testWq);
    }

    ForgeViz.render(document.getElementById('queue-chart'), {
        title: '', chart_type: 'area',
        traces: [
            { x: utils, y: waitTimes, name: 'Wait Time', trace_type: 'area', fill: 'tozeroy', color: '#4a9f6e', width: 2 }
        ],
        reference_lines: [
            { value: rho * 100, axis: 'x', color: '#e74c3c', dash: 'dashed', label: '' }
        ],
        markers: [{ x: rho * 100, y: Wq * 60, label: `Current: ${(Wq * 60).toFixed(1)} min`, color: '#e74c3c' }],
        x_axis: { label: 'Utilization (%)' }, y_axis: { label: 'Wait Time (min)' }
    });

    // Update derivation
    document.getElementById('queue-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Calculate Utilization</div>
            <span class="formula">ρ = λ / (c × μ)</span><br>
            ρ = ${lambda} / (${c} × ${mu}) = ${lambda} / ${c * mu} = <strong>${(rho * 100).toFixed(1)}%</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 2: Calculate P₀ (Empty System Probability)</div>
            <span class="formula">P₀ = 1 / [Σₙ(rⁿ/n!) + (rᶜ/c!)(1/(1-ρ))]</span><br>
            Where r = λ/μ = ${lambda}/${mu} = ${r.toFixed(2)}<br>
            P₀ = <strong>${(P0 * 100).toFixed(2)}%</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 3: Calculate Pᶜ (Erlang C — Probability of Waiting)</div>
            <span class="formula">Pᶜ = (rᶜ/c!) × (1/(1-ρ)) × P₀</span><br>
            Pᶜ = <strong>${(Pc * 100).toFixed(1)}%</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 4: Calculate Queue Length and Wait Time</div>
            <span class="formula">Lq = Pᶜ × ρ / (1-ρ)</span> = ${Lq.toFixed(2)} customers<br>
            <span class="formula">Wq = Lq / λ</span> = ${Lq.toFixed(2)} / ${lambda} = <strong>${(Wq * 60).toFixed(1)} min</strong>
        </div>
        <div class="step">
            <div class="step-num">Insight</div>
            At ${(rho * 100).toFixed(0)}% utilization, ${(Pc * 100).toFixed(0)}% of arrivals must wait.
            ${rho > 0.8 ? '<br><span style="color:#e74c3c">&#9888; High utilization — wait times increase exponentially above 80%</span>' : ''}
        </div>
    `;
}

function toggleQueueMonteCarlo(btn) {
    btn.classList.toggle('active');
    const results = document.getElementById('queue-monte-results');
    results.classList.toggle('visible');

    if (btn.classList.contains('active')) {
        runQueueMonteCarlo();
    }
}

function runQueueMonteCarlo() {
    const lambda = parseFloat(document.getElementById('queue-lambda').value) || 0;
    const mu = parseFloat(document.getElementById('queue-mu').value) || 1;
    const c = parseInt(document.getElementById('queue-c').value) || 1;

    // Queue wait time with variable inputs
    const calcWaitTime = (lam, serv) => {
        const rho = lam / (c * serv);
        if (rho >= 1) return 999; // Unstable

        const r = lam / serv;
        let sum = 0;
        for (let n = 0; n < c; n++) {
            sum += Math.pow(r, n) / factorial(n);
        }
        sum += (Math.pow(r, c) / factorial(c)) * (1 / (1 - rho));
        const P0 = 1 / sum;
        const Pc = (Math.pow(r, c) / factorial(c)) * (1 / (1 - rho)) * P0;
        const Lq = Pc * rho / (1 - rho);
        const Wq = Lq / lam * 60; // in minutes
        return Wq;
    };

    const inputs = [
        { value: lambda, cv: 0.15 },  // ±15% arrival variability
        { value: mu, cv: 0.15 }       // ±15% service variability
    ];

    const sim = MonteCarlo.simulate(calcWaitTime, inputs, 2000);

    // Filter out unstable results
    const validResults = sim.raw.filter(r => r < 900);
    if (validResults.length < 100) {
        document.getElementById('queue-monte-mean').textContent = 'Unstable';
        return;
    }

    document.getElementById('queue-monte-mean').textContent = sim.mean.toFixed(1) + ' min';
    document.getElementById('queue-monte-p5').textContent = sim.p5.toFixed(1) + ' min';
    document.getElementById('queue-monte-p95').textContent = sim.p95.toFixed(1) + ' min';
    document.getElementById('queue-monte-std').textContent = `±${sim.std.toFixed(1)}`;

    MonteCarlo.renderHistogram('queue-monte-chart', sim, 'Wait Time Distribution (2000 runs)', 'min');
}

// ============================================================================
// M/M/c/K Finite Queue
// ============================================================================

function calcQueueFinite() {
    const lambda = parseFloat(document.getElementById('qf-lambda').value) || 0;
    const mu = parseFloat(document.getElementById('qf-mu').value) || 1;
    const c = parseInt(document.getElementById('qf-c').value) || 1;
    const K = parseInt(document.getElementById('qf-k').value) || 10;

    const r = lambda / mu;
    const rho = lambda / (c * mu);

    // Calculate state probabilities P(n) for n = 0 to K
    const P = new Array(K + 1).fill(0);

    // First calculate unnormalized probabilities
    P[0] = 1;
    for (let n = 1; n <= K; n++) {
        if (n < c) {
            P[n] = P[n-1] * r / n;
        } else {
            P[n] = P[n-1] * r / c;
        }
    }

    // Normalize
    const sumP = P.reduce((a, b) => a + b, 0);
    for (let n = 0; n <= K; n++) {
        P[n] /= sumP;
    }

    const Pk = P[K]; // Blocking probability
    const effLambda = lambda * (1 - Pk);
    const lost = lambda * Pk;

    // Calculate L (avg in system)
    let L = 0;
    for (let n = 0; n <= K; n++) {
        L += n * P[n];
    }

    // Calculate Lq (avg in queue)
    let Lq = 0;
    for (let n = c; n <= K; n++) {
        Lq += (n - c) * P[n];
    }

    const W = L / effLambda;
    const Wq = Lq / effLambda;

    document.getElementById('qf-rho').innerHTML = `${(rho * 100).toFixed(1)}<span class="calc-result-unit">%</span>`;
    document.getElementById('qf-wq').innerHTML = `${(Wq * 60).toFixed(1)}<span class="calc-result-unit">min</span>`;
    document.getElementById('qf-lq').textContent = Lq.toFixed(2);
    document.getElementById('qf-pk').innerHTML = `${(Pk * 100).toFixed(1)}<span class="calc-result-unit">%</span>`;
    document.getElementById('qf-eff-lambda').innerHTML = `${effLambda.toFixed(1)}<span class="calc-result-unit">/hr</span>`;
    document.getElementById('qf-lost').textContent = lost.toFixed(2);
    document.getElementById('qf-interpret').textContent =
        Pk > 0.05 ? `&#9888; ${(Pk * 100).toFixed(0)}% of arrivals turned away — consider more capacity` :
        Pk > 0.01 ? `${(Pk * 100).toFixed(1)}% blocking — acceptable for most applications` :
        `<1% blocking — system well-sized`;

    // Update derivation
    document.getElementById('queue-finite-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Traffic Intensity</div>
            <span class="formula">r = λ/μ = ${lambda}/${mu} = ${r.toFixed(2)}</span><br>
            <span class="formula">ρ = λ/(cμ) = ${lambda}/(${c}×${mu}) = ${(rho * 100).toFixed(1)}%</span>
        </div>
        <div class="step">
            <div class="step-num">Step 2: State Probabilities P(0)...P(K)</div>
            M/M/${c}/${K} finite queue — calculate ${K + 1} state probabilities and normalize.<br>
            P₀ = <strong>${(P[0] * 100).toFixed(2)}%</strong> (empty system probability)
        </div>
        <div class="step">
            <div class="step-num">Step 3: Blocking Probability</div>
            <span class="formula">P(K) = probability system is full</span><br>
            P(${K}) = <strong>${(Pk * 100).toFixed(2)}%</strong>
        </div>
        <div class="step">
            <div class="step-num">Step 4: Performance Metrics</div>
            <span class="formula">Effective λ = λ(1 − P_K)</span> = ${lambda} × ${(1 - Pk).toFixed(4)} = <strong>${effLambda.toFixed(2)}/hr</strong><br>
            <span class="formula">L = Σ n×P(n)</span> = <strong>${L.toFixed(2)} in system</strong><br>
            <span class="formula">Lq = Σ (n−c)×P(n) for n≥c</span> = <strong>${Lq.toFixed(2)} in queue</strong><br>
            <span class="formula">W = L/λ_eff</span> = <strong>${(W * 60).toFixed(1)} min in system</strong><br>
            <span class="formula">Wq = Lq/λ_eff</span> = <strong>${(Wq * 60).toFixed(1)} min waiting</strong>
        </div>
    `;

    // Chart: Blocking probability vs capacity
    const caps = [];
    const blockProbs = [];
    for (let testK = c; testK <= Math.max(K + 10, 20); testK++) {
        const testP = calcBlockingProb(lambda, mu, c, testK);
        caps.push(testK);
        blockProbs.push(testP * 100);
    }

    ForgeViz.render(document.getElementById('qf-chart'), {
        title: '', chart_type: 'area',
        traces: [
            { x: caps, y: blockProbs, name: 'Blocking %', trace_type: 'area', fill: 'tozeroy', color: '#e74c3c', width: 2, marker_size: 4 }
        ],
        reference_lines: [
            { value: K, axis: 'x', color: '#4a9f6e', dash: 'dashed', label: '' }
        ],
        markers: [{ x: K, y: Pk * 100, label: `K=${K}: ${(Pk * 100).toFixed(1)}%`, color: '#4a9f6e' }],
        x_axis: { label: 'System Capacity (K)' }, y_axis: { label: 'Blocking Probability (%)' }
    });
}

function calcBlockingProb(lambda, mu, c, K) {
    const r = lambda / mu;
    const P = new Array(K + 1).fill(0);
    P[0] = 1;
    for (let n = 1; n <= K; n++) {
        if (n < c) P[n] = P[n-1] * r / n;
        else P[n] = P[n-1] * r / c;
    }
    const sumP = P.reduce((a, b) => a + b, 0);
    return P[K] / sumP;
}

function toggleQFMonteCarlo(btn) {
    btn.classList.toggle('active');
    document.getElementById('qf-monte-results').classList.toggle('visible');
    if (btn.classList.contains('active')) runQFMonteCarlo();
}

function runQFMonteCarlo() {
    const lambda = parseFloat(document.getElementById('qf-lambda').value) || 0;
    const mu = parseFloat(document.getElementById('qf-mu').value) || 1;
    const c = parseInt(document.getElementById('qf-c').value) || 1;
    const K = parseInt(document.getElementById('qf-k').value) || 10;

    const calcBlocking = (lam, serv) => calcBlockingProb(lam, serv, c, K) * 100;
    const inputs = [
        { value: lambda, cv: 0.15 },
        { value: mu, cv: 0.15 }
    ];
    const sim = MonteCarlo.simulate(calcBlocking, inputs, 2000);

    document.getElementById('qf-monte-mean').textContent = sim.mean.toFixed(1) + '%';
    document.getElementById('qf-monte-p95').textContent = sim.p95.toFixed(1) + '%';
    document.getElementById('qf-monte-max').textContent = sim.max.toFixed(1) + '%';
    MonteCarlo.renderHistogram('qf-monte-chart', sim, 'Blocking % Distribution', '%');
}

// ============================================================================
// Priority Queue
// ============================================================================

let priorityClasses = [
    { name: 'High (Code Red)', lambda: 2, color: '#e74c3c' },
    { name: 'Medium', lambda: 5, color: '#f39c12' },
    { name: 'Low (Routine)', lambda: 8, color: '#27ae60' }
];

function renderPriorityClasses() {
    const container = document.getElementById('qp-classes');
    container.innerHTML = priorityClasses.map((pc, i) => `
        <div style="display: flex; align-items: center; gap: 12px; padding: 12px; background: var(--bg-secondary); border-radius: 8px; border-left: 4px solid ${pc.color};">
            <input type="text" value="${pc.name}" style="flex: 1; padding: 8px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary);" onchange="updatePriorityClass(${i}, 'name', this.value)">
            <div style="display: flex; align-items: center; gap: 4px;">
                <span style="font-size: 12px; color: var(--text-dim);">λ:</span>
                <input type="number" value="${pc.lambda}" style="width: 60px; padding: 8px; text-align: center; background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary);" oninput="updatePriorityClass(${i}, 'lambda', this.value)">
                <span style="font-size: 11px; color: var(--text-dim);">/hr</span>
            </div>
            <input type="color" value="${pc.color}" style="width: 32px; height: 32px; border: none; cursor: pointer;" onchange="updatePriorityClass(${i}, 'color', this.value)">
            ${priorityClasses.length > 2 ? `<button onclick="removePriorityClass(${i})" style="padding: 4px 8px; background: transparent; border: none; color: var(--text-dim); cursor: pointer;">&times;</button>` : ''}
        </div>
    `).join('');
    calcQueuePriority();
}

function addPriorityClass() {
    priorityClasses.push({ name: `Class ${priorityClasses.length + 1}`, lambda: 3, color: '#3498db' });
    renderPriorityClasses();
}

function removePriorityClass(i) {
    priorityClasses.splice(i, 1);
    renderPriorityClasses();
}

function updatePriorityClass(i, field, value) {
    if (field === 'lambda') value = parseFloat(value) || 0;
    priorityClasses[i][field] = value;
    calcQueuePriority();
}

function calcQueuePriority() {
    const c = parseInt(document.getElementById('qp-c').value) || 1;
    const mu = parseFloat(document.getElementById('qp-mu').value) || 1;

    const totalLambda = priorityClasses.reduce((a, pc) => a + pc.lambda, 0);
    const rho = totalLambda / (c * mu);

    if (rho >= 1) {
        document.getElementById('qp-results').innerHTML = '<p style="color: #e74c3c;">System unstable (total λ ≥ cμ)</p>';
        return;
    }

    // Calculate wait times for each priority class (non-preemptive)
    const results = [];
    let cumulativeRho = 0;

    for (let k = 0; k < priorityClasses.length; k++) {
        const pc = priorityClasses[k];
        const rhoK = pc.lambda / (c * mu);
        const prevCumulativeRho = cumulativeRho;
        cumulativeRho += rhoK;

        // Wait time for priority class k (non-preemptive priority)
        // Wq_k = W0 / ((1 - sum_j<k rho_j)(1 - sum_j<=k rho_j))
        // where W0 is base wait for M/M/c
        const baseWq = calcMMcWait(totalLambda, mu, c);
        const denom = (1 - prevCumulativeRho) * (1 - cumulativeRho);
        const Wq = denom > 0 ? baseWq / denom * rhoK / (rho) : 999;

        results.push({
            name: pc.name,
            color: pc.color,
            lambda: pc.lambda,
            Wq: Wq * 60, // minutes
            share: (pc.lambda / totalLambda * 100)
        });
    }

    // Render results table
    document.getElementById('qp-results').innerHTML = `
        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
            <tr style="border-bottom: 1px solid var(--border);">
                <th style="padding: 10px; text-align: left;">Priority Class</th>
                <th style="padding: 10px; text-align: right;">Arrival Rate</th>
                <th style="padding: 10px; text-align: right;">% of Traffic</th>
                <th style="padding: 10px; text-align: right;">Avg Wait</th>
            </tr>
            ${results.map(r => `
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 10px;"><span style="display: inline-block; width: 12px; height: 12px; background: ${r.color}; border-radius: 2px; margin-right: 8px;"></span>${r.name}</td>
                    <td style="padding: 10px; text-align: right;">${r.lambda}/hr</td>
                    <td style="padding: 10px; text-align: right;">${r.share.toFixed(0)}%</td>
                    <td style="padding: 10px; text-align: right; font-weight: 600; color: ${r.Wq > 30 ? '#e74c3c' : r.Wq > 10 ? '#f39c12' : '#27ae60'};">${r.Wq.toFixed(1)} min</td>
                </tr>
            `).join('')}
        </table>
    `;

    // Chart
    ForgeViz.render(document.getElementById('qp-chart'), {
        title: '', chart_type: 'bar',
        traces: [
            { x: results.map(r => r.name), y: results.map(r => r.Wq), name: 'Wait Time', trace_type: 'bar', color: results.map(r => r.color) }
        ],
        x_axis: { label: '' }, y_axis: { label: 'Average Wait (min)' }
    });

    // Update derivation
    const prioritySteps = results.map((r, i) => `
        <div class="step">
            <div class="step-num">Class ${i + 1}: ${r.name}</div>
            λ = ${r.lambda}/hr (${r.share.toFixed(0)}% of traffic)<br>
            Average wait: <strong>${r.Wq.toFixed(1)} min</strong>
        </div>`).join('');
    document.getElementById('queue-priority-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">System Parameters</div>
            Total λ = ${totalLambda}/hr, μ = ${mu}/hr, c = ${c} server(s)<br>
            <span class="formula">ρ = λ/(cμ) = ${totalLambda}/(${c}×${mu}) = ${(rho * 100).toFixed(1)}%</span>
        </div>
        <div class="step">
            <div class="step-num">Priority Queue Model</div>
            Non-preemptive priority: higher-priority classes get shorter waits at the expense of lower-priority classes.<br>
            <span class="formula">Wq_k = W₀ × ρ_k / (ρ × (1−σ_{k-1})(1−σ_k))</span>
        </div>
        ${prioritySteps}
    `;
}

function calcMMcWait(lambda, mu, c) {
    const rho = lambda / (c * mu);
    if (rho >= 1) return 999;
    const r = lambda / mu;
    let sum = 0;
    for (let n = 0; n < c; n++) sum += Math.pow(r, n) / factorial(n);
    sum += (Math.pow(r, c) / factorial(c)) * (1 / (1 - rho));
    const P0 = 1 / sum;
    const Pc = (Math.pow(r, c) / factorial(c)) * (1 / (1 - rho)) * P0;
    const Lq = Pc * rho / (1 - rho);
    return Lq / lambda;
}

// ============================================================================
// Staffing Optimizer
// ============================================================================

function calcOptimizer() {
    const lambda = parseFloat(document.getElementById('qo-lambda').value) || 0;
    const mu = parseFloat(document.getElementById('qo-mu').value) || 1;
    const serverCost = parseFloat(document.getElementById('qo-server-cost').value) || 0;
    const waitCost = parseFloat(document.getElementById('qo-wait-cost').value) || 0;
    const targetWait = parseFloat(document.getElementById('qo-target-wait').value) || null;

    const minServers = Math.ceil(lambda / mu);
    const maxServers = minServers + 10;

    const data = [];
    let optimalC = minServers;
    let minTotalCost = Infinity;

    for (let c = minServers; c <= maxServers; c++) {
        const rho = lambda / (c * mu);
        if (rho >= 1) continue;

        const Wq = calcMMcWait(lambda, mu, c);
        const WqMin = Wq * 60;
        const staffCostHr = c * serverCost;
        const waitCostHr = lambda * Wq * waitCost;
        const totalCost = staffCostHr + waitCostHr;

        const meetsTarget = targetWait ? WqMin <= targetWait : true;

        data.push({ c, rho: rho * 100, WqMin, staffCostHr, waitCostHr, totalCost, meetsTarget });

        if (totalCost < minTotalCost && meetsTarget) {
            minTotalCost = totalCost;
            optimalC = c;
        }
    }

    const optimal = data.find(d => d.c === optimalC);
    document.getElementById('qo-optimal').textContent = optimalC;
    document.getElementById('qo-total-cost').textContent = '$' + Math.round(optimal?.totalCost || 0);
    document.getElementById('qo-wait').innerHTML = `${(optimal?.WqMin || 0).toFixed(1)}<span class="calc-result-unit">min</span>`;
    document.getElementById('qo-util').innerHTML = `${(optimal?.rho || 0).toFixed(0)}<span class="calc-result-unit">%</span>`;

    // Update derivation
    document.getElementById('queue-optimizer-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Step 1: Minimum Servers Required</div>
            <span class="formula">c_min = ⌈λ/μ⌉ = ⌈${lambda}/${mu}⌉ = ${minServers}</span><br>
            System requires at least ${minServers} server(s) for stability.
        </div>
        <div class="step">
            <div class="step-num">Step 2: Cost Model</div>
            <span class="formula">Total Cost = (c × Server Cost) + (λ × Wq × Wait Cost)</span><br>
            Server cost = $${serverCost}/hr per server, Wait cost = $${waitCost}/hr per waiting customer
        </div>
        <div class="step">
            <div class="step-num">Step 3: Optimal Staffing</div>
            Tested c = ${minServers} to ${maxServers} servers<br>
            Optimal: <strong>c = ${optimalC}</strong> at $${Math.round(optimal?.totalCost || 0)}/hr total<br>
            (Staff: $${Math.round(optimal?.staffCostHr || 0)}/hr + Wait: $${Math.round(optimal?.waitCostHr || 0)}/hr)
        </div>
        <div class="step">
            <div class="step-num">Performance at Optimal</div>
            Utilization: <strong>${(optimal?.rho || 0).toFixed(0)}%</strong>,
            Avg Wait: <strong>${(optimal?.WqMin || 0).toFixed(1)} min</strong>
        </div>
    `;

    // Chart
    ForgeViz.render(document.getElementById('qo-chart'), {
        title: '', chart_type: 'stacked_bar',
        traces: [
            { x: data.map(d => d.c), y: data.map(d => d.staffCostHr), name: 'Server Cost', trace_type: 'bar', color: '#3a7f8f' },
            { x: data.map(d => d.c), y: data.map(d => d.waitCostHr), name: 'Wait Cost', trace_type: 'bar', color: '#e89547' },
            { x: data.map(d => d.c), y: data.map(d => d.totalCost), name: 'Total Cost', trace_type: 'line', color: '#4a9f6e', width: 3 }
        ],
        reference_lines: [
            { value: optimalC, axis: 'x', color: '#e8c547', dash: 'dashed', label: '' }
        ],
        markers: [{ x: optimalC, y: optimal?.totalCost, label: `Optimal: ${optimalC} servers`, color: '#e8c547' }],
        x_axis: { label: 'Number of Servers' }, y_axis: { label: 'Cost Component ($/hr)' }
    });

    // Table
    document.getElementById('qo-table').innerHTML = `
        <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
            <tr style="border-bottom: 2px solid var(--border); background: var(--bg-secondary);">
                <th style="padding: 10px; text-align: center;">Servers</th>
                <th style="padding: 10px; text-align: right;">Utilization</th>
                <th style="padding: 10px; text-align: right;">Avg Wait</th>
                <th style="padding: 10px; text-align: right;">Server Cost</th>
                <th style="padding: 10px; text-align: right;">Wait Cost</th>
                <th style="padding: 10px; text-align: right;">Total</th>
            </tr>
            ${data.map(d => `
                <tr style="border-bottom: 1px solid var(--border); ${d.c === optimalC ? 'background: rgba(232, 197, 71, 0.1);' : ''}">
                    <td style="padding: 10px; text-align: center; font-weight: ${d.c === optimalC ? '700' : '400'};">${d.c === optimalC ? '★ ' : ''}${d.c}</td>
                    <td style="padding: 10px; text-align: right; color: ${d.rho > 90 ? '#e74c3c' : d.rho > 80 ? '#f39c12' : '#27ae60'};">${d.rho.toFixed(0)}%</td>
                    <td style="padding: 10px; text-align: right; ${!d.meetsTarget ? 'color: #e74c3c;' : ''}">${d.WqMin.toFixed(1)} min${!d.meetsTarget ? ' &#9888;' : ''}</td>
                    <td style="padding: 10px; text-align: right;">$${d.staffCostHr.toFixed(0)}</td>
                    <td style="padding: 10px; text-align: right;">$${d.waitCostHr.toFixed(0)}</td>
                    <td style="padding: 10px; text-align: right; font-weight: 600;">$${d.totalCost.toFixed(0)}</td>
                </tr>
            `).join('')}
        </table>
    `;
}

// Queue Simulator + A/B Compare — see calc-sim-flow.js

// ============================================================================
// Multi-Stage (Tandem) Queue
// ============================================================================

let tandemStages = [
    { name: 'Triage', mu: 20, c: 1, color: '#e74c3c' },
    { name: 'Doctor', mu: 3, c: 2, color: '#3498db' },
    { name: 'Checkout', mu: 12, c: 1, color: '#27ae60' }
];

function renderTandemStages() {
    const container = document.getElementById('qt-stages');
    container.innerHTML = tandemStages.map((stage, i) => `
        <div style="display: flex; align-items: center; gap: 12px; padding: 16px; background: var(--bg-secondary); border-radius: 8px; border-left: 4px solid ${stage.color};">
            <div style="flex: 1;">
                <input type="text" value="${stage.name}" style="font-size: 14px; font-weight: 600; padding: 6px 10px; background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary); width: 120px;" onchange="updateTandemStage(${i}, 'name', this.value)">
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 11px; color: var(--text-dim);">μ:</span>
                <input type="number" value="${stage.mu}" style="width: 50px; padding: 6px; text-align: center; background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary);" oninput="updateTandemStage(${i}, 'mu', this.value)">
                <span style="font-size: 10px; color: var(--text-dim);">/hr/server</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 11px; color: var(--text-dim);">Servers:</span>
                <input type="number" value="${stage.c}" min="1" style="width: 50px; padding: 6px; text-align: center; background: var(--bg-card); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary);" oninput="updateTandemStage(${i}, 'c', this.value)">
            </div>
            <input type="color" value="${stage.color}" style="width: 32px; height: 32px; border: none; cursor: pointer; border-radius: 4px;" onchange="updateTandemStage(${i}, 'color', this.value)">
            ${tandemStages.length > 2 ? `<button onclick="removeTandemStage(${i})" style="padding: 4px 8px; background: transparent; border: none; color: var(--text-dim); cursor: pointer; font-size: 18px;">&times;</button>` : ''}
            ${i < tandemStages.length - 1 ? '<span style="font-size: 20px; color: var(--text-dim);">→</span>' : ''}
        </div>
    `).join('');
    calcTandem();
}

function addTandemStage() {
    tandemStages.push({ name: `Stage ${tandemStages.length + 1}`, mu: 6, c: 1, color: '#9b59b6' });
    renderTandemStages();
}

function removeTandemStage(i) {
    tandemStages.splice(i, 1);
    renderTandemStages();
}

function updateTandemStage(i, field, value) {
    if (field === 'mu' || field === 'c') value = parseFloat(value) || 1;
    tandemStages[i][field] = value;
    calcTandem();
}

function calcTandem() {
    const lambda = parseFloat(document.getElementById('qt-lambda').value) || 10;

    // Calculate metrics for each stage
    const results = tandemStages.map(stage => {
        const rho = lambda / (stage.c * stage.mu);
        if (rho >= 1) return { ...stage, rho, Wq: 999, W: 999, stable: false };

        const Wq = calcMMcWait(lambda, stage.mu, stage.c);
        const W = Wq + (1 / stage.mu);

        return { ...stage, rho, Wq: Wq * 60, W: W * 60, stable: true };
    });

    // Find bottleneck (highest utilization)
    const bottleneck = results.reduce((max, r) => r.rho > max.rho ? r : max, results[0]);

    // Total times
    const totalWait = results.reduce((sum, r) => sum + r.Wq, 0);
    const totalTime = results.reduce((sum, r) => sum + r.W, 0);

    document.getElementById('qt-total-time').innerHTML = `${totalTime.toFixed(0)}<span class="calc-result-unit">min</span>`;
    document.getElementById('qt-total-wait').innerHTML = `${totalWait.toFixed(0)}<span class="calc-result-unit">min</span>`;
    document.getElementById('qt-bottleneck').innerHTML = `<span style="color: ${bottleneck.color};">${bottleneck.name}</span>`;

    // Breakdown table
    document.getElementById('qt-breakdown').innerHTML = `
        <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
            <tr style="border-bottom: 2px solid var(--border); background: var(--bg-secondary);">
                <th style="padding: 10px; text-align: left;">Stage</th>
                <th style="padding: 10px; text-align: right;">Servers</th>
                <th style="padding: 10px; text-align: right;">Utilization</th>
                <th style="padding: 10px; text-align: right;">Wait Time</th>
                <th style="padding: 10px; text-align: right;">Service Time</th>
                <th style="padding: 10px; text-align: right;">Total at Stage</th>
            </tr>
            ${results.map(r => `
                <tr style="border-bottom: 1px solid var(--border); ${r.name === bottleneck.name ? 'background: rgba(231, 76, 60, 0.1);' : ''}">
                    <td style="padding: 10px;"><span style="display: inline-block; width: 12px; height: 12px; background: ${r.color}; border-radius: 2px; margin-right: 8px;"></span>${r.name}${r.name === bottleneck.name ? ' <span style="color:#e74c3c;font-weight:700;">&#9679;</span>' : ''}</td>
                    <td style="padding: 10px; text-align: right;">${r.c}</td>
                    <td style="padding: 10px; text-align: right; color: ${r.rho > 0.9 ? '#e74c3c' : r.rho > 0.8 ? '#f39c12' : '#27ae60'}; font-weight: 600;">${r.stable ? (r.rho * 100).toFixed(0) + '%' : '&#9888; Unstable'}</td>
                    <td style="padding: 10px; text-align: right;">${r.Wq.toFixed(1)} min</td>
                    <td style="padding: 10px; text-align: right;">${(60 / r.mu).toFixed(1)} min</td>
                    <td style="padding: 10px; text-align: right; font-weight: 600;">${r.W.toFixed(1)} min</td>
                </tr>
            `).join('')}
            <tr style="background: var(--bg-secondary); font-weight: 600;">
                <td style="padding: 10px;" colspan="3">TOTAL END-TO-END</td>
                <td style="padding: 10px; text-align: right;">${totalWait.toFixed(0)} min</td>
                <td style="padding: 10px; text-align: right;">${results.reduce((s, r) => s + 60/r.mu, 0).toFixed(0)} min</td>
                <td style="padding: 10px; text-align: right; color: var(--accent);">${totalTime.toFixed(0)} min</td>
            </tr>
        </table>
    `;

    // Chart - stacked bar
    ForgeViz.render(document.getElementById('qt-chart'), {
        title: '', chart_type: 'stacked_bar',
        traces: [
            { x: results.map(r => r.name), y: results.map(r => r.Wq), name: 'Wait Time', trace_type: 'bar', color: 'rgba(231, 76, 60, 0.7)' },
            { x: results.map(r => r.name), y: results.map(r => 60 / r.mu), name: 'Service Time', trace_type: 'bar', color: results.map(r => r.color) }
        ],
        x_axis: { label: '' }, y_axis: { label: 'Time (min)' }
    });

    // Update derivation
    const tandemSteps = results.map(r => `
        <div class="step">
            <div class="step-num">${r.name} (c=${r.c}, μ=${r.mu}/hr)</div>
            ρ = ${lambda}/(${r.c}×${r.mu}) = <strong>${(r.rho * 100).toFixed(0)}%</strong>${r.name === bottleneck.name ? ' ← Bottleneck' : ''}<br>
            Wait: ${r.Wq.toFixed(1)} min + Service: ${(60/r.mu).toFixed(1)} min = <strong>${r.W.toFixed(1)} min</strong>
        </div>`).join('');
    document.getElementById('queue-tandem-derivation-body').innerHTML = `
        <div class="step">
            <div class="step-num">Tandem Queue Model</div>
            Jackson's theorem: each stage analyzed independently as M/M/c with arrival rate λ = ${lambda}/hr.
        </div>
        ${tandemSteps}
        <div class="step">
            <div class="step-num">End-to-End</div>
            <span class="formula">Total Time = Σ (Wait + Service) across all stages</span><br>
            = <strong>${totalTime.toFixed(0)} min</strong> (${totalWait.toFixed(0)} min waiting + ${results.reduce((s, r) => s + 60/r.mu, 0).toFixed(0)} min service)
        </div>
    `;
}

function toggleTandemMonteCarlo(btn) {
    btn.classList.toggle('active');
    document.getElementById('qt-monte-results').classList.toggle('visible');
    if (btn.classList.contains('active')) runTandemMonteCarlo();
}

function runTandemMonteCarlo() {
    const lambda = parseFloat(document.getElementById('qt-lambda').value) || 10;

    const calcTotalTime = (...mus) => {
        let total = 0;
        tandemStages.forEach((stage, i) => {
            const mu = mus[i] || stage.mu;
            const rho = lambda / (stage.c * mu);
            if (rho >= 1) return 999;
            const Wq = calcMMcWait(lambda, mu, stage.c);
            const W = Wq + (1 / mu);
            total += W * 60;
        });
        return total;
    };

    const inputs = tandemStages.map(s => ({ value: s.mu, cv: 0.15 }));
    const sim = MonteCarlo.simulate(calcTotalTime, inputs, 2000);

    document.getElementById('qt-monte-mean').textContent = sim.mean.toFixed(0) + ' min';
    document.getElementById('qt-monte-p5').textContent = sim.p5.toFixed(0) + ' min';
    document.getElementById('qt-monte-p95').textContent = sim.p95.toFixed(0) + ' min';
    MonteCarlo.renderHistogram('qt-monte-chart', sim, 'Total Time Distribution', 'min');
}
