// =============================================================================
// Interaction — Select, Drag, Connect, Delete
// =============================================================================

function selectElement(el, type) {
    if (currentTool === 'connect') {
        if (!connectFrom) {
            connectFrom = el;
        } else {
            if (connectFrom.id !== el.id) {
                layout.connections.push({
                    id: genId('conn'),
                    from_id: connectFrom.id,
                    to_id: el.id,
                    buffer_capacity: null,
                    transport_type: 'none',
                    transport_distance: 0,
                });
            }
            connectFrom = null;
            renderCanvas();
        }
        return;
    }

    selectedElement = el;
    selectedElement._type = type;
    showProperties(el, type);
    renderCanvas();
}

function showProperties(el, type) {
    const panel = document.getElementById('props-panel');
    const content = document.getElementById('props-content');
    panel.classList.add('active');

    if (type === 'machine') {
        content.innerHTML = `
            <div class="prop-row"><span class="prop-label">Name</span><input class="prop-input" value="${el.name || ''}" onchange="updateProp('name', this.value)"></div>
            <div class="prop-row"><span class="prop-label">Cycle Time</span><input type="number" class="prop-input" value="${el.cycle_time || 30}" onchange="updateProp('cycle_time', +this.value)"><span class="prop-unit">s</span></div>
            <div class="prop-row"><span class="prop-label">CV %</span><input type="number" class="prop-input" value="${((el.cycle_time_cv || 0) * 100).toFixed(0)}" onchange="updateProp('cycle_time_cv', +this.value/100)" step="5"><span class="prop-unit">%</span></div>
            <div class="prop-row"><span class="prop-label">C/O Default</span><input type="number" class="prop-input" value="${el.changeover_time || 0}" onchange="updateProp('changeover_time', +this.value)" title="Default changeover time when switching product types"><span class="prop-unit">s</span></div>
            <div style="margin-top:2px; margin-bottom:6px;">
                <span class="prop-label" style="width:auto; display:block; margin-bottom:2px; font-size:0.6rem;">Setup Matrix (sequence-dependent)</span>
                ${renderSetupMatrixHTML(el)}
            </div>
            <div class="prop-row"><span class="prop-label">Uptime</span><input type="number" class="prop-input" value="${el.uptime || 100}" onchange="updateProp('uptime', +this.value)"><span class="prop-unit">%</span></div>
            <div class="prop-row"><span class="prop-label">MTBF</span><input type="number" class="prop-input" value="${el.mtbf || ''}" onchange="updateProp('mtbf', +this.value || null)" placeholder="sec"><span class="prop-unit">s</span></div>
            <div class="prop-row"><span class="prop-label">MTTR</span><input type="number" class="prop-input" value="${el.mttr || ''}" onchange="updateProp('mttr', +this.value || null)" placeholder="sec"><span class="prop-unit">s</span></div>
            <div class="prop-row"><span class="prop-label">Weibull β</span><input type="number" class="prop-input" value="${el.weibull_beta || 1}" onchange="updateProp('weibull_beta', +this.value)" step="0.1" min="0.1" max="5" title="Shape: 1=random (exponential), >1=wear-out, <1=infant mortality"><span class="prop-unit"></span></div>
            <div style="font-size:0.5rem;color:var(--text-dim);margin:-2px 0 4px 76px;">1=random &nbsp; 2=wear-out &nbsp; 0.5=infant</div>
            <div class="prop-row"><span class="prop-label">PM Interval</span><input type="number" class="prop-input" value="${el.pm_interval || ''}" onchange="updateProp('pm_interval', +this.value || 0)" placeholder="disabled"><span class="prop-unit">s</span></div>
            <div class="prop-row"><span class="prop-label">PM Duration</span><input type="number" class="prop-input" value="${el.pm_duration || ''}" onchange="updateProp('pm_duration', +this.value || 0)" placeholder="0"><span class="prop-unit">s</span></div>
            <div class="prop-row"><span class="prop-label">Detection %</span><input type="number" class="prop-input" value="${((el.defect_detection_rate ?? 1) * 100).toFixed(0)}" onchange="updateProp('defect_detection_rate', +this.value/100)" step="5" min="0" max="100" title="Probability defects are caught at this station. Undetected defects escape downstream."><span class="prop-unit">%</span></div>
            <div class="prop-row"><span class="prop-label">Drift Rate</span><input type="number" class="prop-input" value="${((el.drift_rate || 0) * 100).toFixed(1)}" onchange="updateProp('drift_rate', +this.value/100)" step="0.1" min="0" title="Scrap rate increase per 100 jobs. Process slowly goes out of spec."><span class="prop-unit">%/100j</span></div>
            <div class="prop-row"><span class="prop-label">Auto-Cal @</span><input type="number" class="prop-input" value="${el.calibration_interval || ''}" onchange="updateProp('calibration_interval', +this.value || 0)" placeholder="manual" min="0" title="Auto-calibrate every N jobs (resets drift). 0 = never."><span class="prop-unit">jobs</span></div>
            <div class="prop-row"><span class="prop-label">Handover %</span><input type="number" class="prop-input" value="${((el.handover_loss_rate || 0) * 100).toFixed(0)}" onchange="updateProp('handover_loss_rate', +this.value/100)" step="5" min="0" max="100" title="Probability of repeated setup at shift change (crew forgot what was running)"><span class="prop-unit">%</span></div>
            <div class="prop-row"><span class="prop-label">Warmup</span><input type="number" class="prop-input" value="${el.warmup_time || ''}" onchange="updateProp('warmup_time', +this.value || 0)" placeholder="0" min="0" title="Seconds of warmup needed after idle period"><span class="prop-unit">s</span></div>
            <div class="prop-row"><span class="prop-label">Contam %</span><input type="number" class="prop-input" value="${((el.contamination_risk || 0) * 100).toFixed(0)}" onchange="updateProp('contamination_risk', +this.value/100)" step="5" min="0" max="100" title="Contamination risk after quick changeover. Residue from previous product causes hidden defects."><span class="prop-unit">%</span></div>
            <div class="prop-row"><span class="prop-label">μ-Stop %</span><input type="number" class="prop-input" value="${((el.micro_stop_rate || 0) * 100).toFixed(0)}" onchange="updateProp('micro_stop_rate', +this.value/100)" step="5" min="0" max="100" title="Micro-stoppage probability per cycle. Brief jams/sensor trips that hide in processing time."><span class="prop-unit">%</span></div>
            <div class="prop-row"><span class="prop-label">μ-Stop Dur</span><input type="number" class="prop-input" value="${el.micro_stop_duration || 10}" onchange="updateProp('micro_stop_duration', +this.value)" min="1" title="Average micro-stop duration (seconds)"><span class="prop-unit">s</span></div>
            <div class="prop-row"><span class="prop-label">1st Article</span><input type="number" class="prop-input" value="${((el.first_article_penalty || 0) * 100).toFixed(0)}" onchange="updateProp('first_article_penalty', +this.value/100)" step="5" min="0" max="100" title="Extra scrap % on first parts after restart"><span class="prop-unit">%</span></div>
            <div class="prop-row"><span class="prop-label">Dispatch</span>
                <select class="prop-input" onchange="updateProp('dispatch_rule', this.value)" title="Queue dispatch rule: how the next job is selected from the queue">
                    <option value="FIFO" ${(el.dispatch_rule || 'FIFO') === 'FIFO' ? 'selected' : ''}>FIFO</option>
                    <option value="SPT" ${el.dispatch_rule === 'SPT' ? 'selected' : ''}>SPT</option>
                    <option value="EDD" ${el.dispatch_rule === 'EDD' ? 'selected' : ''}>EDD</option>
                    <option value="CR" ${el.dispatch_rule === 'CR' ? 'selected' : ''}>CR</option>
                    <option value="WSPT" ${el.dispatch_rule === 'WSPT' ? 'selected' : ''}>WSPT</option>
                </select>
            </div>
            <div style="font-size:0.5rem;color:var(--text-dim);margin:-2px 0 4px 76px;">FIFO=first in &nbsp; SPT=shortest &nbsp; EDD=due date &nbsp; CR=critical ratio</div>
            <div class="prop-row"><span class="prop-label">Dedicated</span>
                <select class="prop-input" onchange="updateProp('dedicated_product', this.value || null)" title="Restrict this machine to one product type. 'Any' = processes all types.">
                    <option value="" ${!el.dedicated_product ? 'selected' : ''}>Any product</option>
                    ${getAllProductTypes().map(pt =>
                        `<option value="${pt}" ${el.dedicated_product === pt ? 'selected' : ''}>${pt}</option>`
                    ).join('')}
                </select>
            </div>
            <div class="prop-row"><span class="prop-label">Operators</span><input type="number" class="prop-input" value="${el.operators || 1}" onchange="updateProp('operators', +this.value)" min="1"></div>
            <div class="prop-row"><span class="prop-label">Transfer Batch</span><input type="number" class="prop-input" value="${el.batch_size || 1}" onchange="updateProp('batch_size', +this.value)" min="1" title="Min parts in queue before machine starts processing"></div>
            <div class="prop-row"><span class="prop-label">Base Scrap %</span><input type="number" class="prop-input" value="${((el.scrap_rate || 0) * 100).toFixed(0)}" onchange="updateProp('scrap_rate', +this.value/100)" step="1" min="0" max="100" title="Material waste (part destroyed, independent of defects)"><span class="prop-unit">%</span></div>
            <div class="prop-row"><span class="prop-label">Defect %</span><input type="number" class="prop-input" value="${((el.defect_rate || 0) * 100).toFixed(0)}" onchange="updateProp('defect_rate', +this.value/100)" step="1" min="0" max="100" title="Probability of nonconformance (separate from scrap). Enables 3-stage quality model."><span class="prop-unit">%</span></div>
            <div class="prop-row"><span class="prop-label">Rework %</span><input type="number" class="prop-input" value="${((el.rework_rate || 0) * 100).toFixed(0)}" onchange="updateProp('rework_rate', +this.value/100)" step="1" min="0" max="100" title="${(el.defect_rate || 0) > 0 ? 'Probability a detected defect is reworkable (vs scrapped)' : 'Flat rework probability (legacy mode)'}"><span class="prop-unit">%</span></div>
            <div class="prop-row"><span class="prop-label">Rework →</span>
                <select class="prop-input" onchange="updateProp('rework_target', this.value || null)" title="Where rework jobs go. Self = back into this machine's queue.">
                    <option value="" ${!el.rework_target ? 'selected' : ''}>Self</option>
                    ${layout.stations.filter(s => s.id !== el.id).map(s =>
                        `<option value="${s.id}" ${el.rework_target === s.id ? 'selected' : ''}>${s.name || s.id}</option>`
                    ).join('')}
                </select>
            </div>
            <div style="margin-top:4px; border-top:1px solid var(--border); padding-top:4px;">
                <span class="prop-label" style="width:auto; display:block; margin-bottom:2px; font-size:0.6rem;">Quality by Product <span style="color:var(--text-dim)">(blank = use defaults)</span></span>
                ${getAllProductTypes().length === 0
                    ? '<div style="font-size:0.5rem;color:var(--text-dim);">Add product sources first</div>'
                    : getAllProductTypes().map(pt => {
                        const _qp = (el.quality_by_product || {})[pt] || {};
                        return `<div style="font-size:0.55rem; margin-bottom:4px;">
                            <strong>${pt}</strong>
                            <div style="display:flex; gap:4px; align-items:center; margin-top:1px;">
                                <span style="font-size:0.5rem;color:var(--text-dim);">Scrap</span>
                                <input type="number" class="prop-input" value="${_qp.scrap_rate != null ? (_qp.scrap_rate * 100).toFixed(0) : ''}"
                                    onchange="updateQualityByProduct('${pt}', 'scrap_rate', this.value)"
                                    placeholder="—" step="1" min="0" max="100" style="width:36px; padding:1px 2px; font-size:0.55rem;">
                                <span style="font-size:0.5rem;color:var(--text-dim);">Def</span>
                                <input type="number" class="prop-input" value="${_qp.defect_rate != null ? (_qp.defect_rate * 100).toFixed(0) : ''}"
                                    onchange="updateQualityByProduct('${pt}', 'defect_rate', this.value)"
                                    placeholder="—" step="1" min="0" max="100" style="width:36px; padding:1px 2px; font-size:0.55rem;">
                                <span style="font-size:0.5rem;color:var(--text-dim);">Rwk</span>
                                <input type="number" class="prop-input" value="${_qp.rework_rate != null ? (_qp.rework_rate * 100).toFixed(0) : ''}"
                                    onchange="updateQualityByProduct('${pt}', 'rework_rate', this.value)"
                                    placeholder="—" step="1" min="0" max="100" style="width:36px; padding:1px 2px; font-size:0.55rem;">
                            </div>
                        </div>`;
                    }).join('')
                }
            </div>
            <div class="prop-row"><span class="prop-label">Shift</span>
                <select class="prop-input" onchange="updateProp('shift_schedule', this.value)">
                    <option value="24/7" ${(el.shift_schedule || '24/7') === '24/7' ? 'selected' : ''}>24/7</option>
                    <option value="single-8" ${el.shift_schedule === 'single-8' ? 'selected' : ''}>1 x 8hr</option>
                    <option value="double-16" ${el.shift_schedule === 'double-16' ? 'selected' : ''}>2 x 8hr</option>
                    <option value="triple-24" ${el.shift_schedule === 'triple-24' ? 'selected' : ''}>3 x 8hr</option>
                    <option value="single-12" ${el.shift_schedule === 'single-12' ? 'selected' : ''}>1 x 12hr</option>
                </select>
            </div>
            <div style="margin-top:6px; border-top:1px solid var(--border); padding-top:6px;">
                <span class="prop-label" style="width:auto; display:block; margin-bottom:2px; font-size:0.6rem;">Inline SPC</span>
                <div class="prop-row">
                    <span class="prop-label">SPC</span>
                    <select class="prop-input" onchange="updateProp('spc_enabled', this.value === 'true')">
                        <option value="false" ${!el.spc_enabled ? 'selected' : ''}>Off</option>
                        <option value="true" ${el.spc_enabled ? 'selected' : ''}>On</option>
                    </select>
                </div>
                <div class="prop-row"><span class="prop-label">Investigate</span><input type="number" class="prop-input" value="${el.spc_investigation_time || 120}" onchange="updateProp('spc_investigation_time', +this.value)" min="0" title="Seconds the machine stops when SPC signals out-of-control"><span class="prop-unit">s</span></div>
                <div class="prop-row"><span class="prop-label">Meas. Lag</span><input type="number" class="prop-input" value="${el.measurement_delay || 0}" onchange="updateProp('measurement_delay', +this.value)" min="0" step="30" title="Seconds before a measurement result appears on the SPC chart. Simulates lab turnaround, CMM queue, etc. Parts keep flowing while you wait for data."><span class="prop-unit">s</span></div>
                <div style="font-size:0.5rem;color:var(--text-dim);line-height:1.3;margin-top:2px;">Western Electric rules. Tight SPC = more false alarms but catches drift early. Loose = fewer stops but more escapes. Measurement lag creates containment scope.</div>
            </div>
            <div style="margin-top:6px; border-top:1px solid var(--border); padding-top:6px;">
                <div class="prop-row">
                    <span class="prop-label">Accumulation</span>
                    <select class="prop-input" onchange="updateProp('batch_process_mode', this.value === 'true'); showProperties(selectedElement, 'machine');" title="Oven/furnace mode: accumulate parts, process all at once">
                        <option value="false" ${!el.batch_process_mode ? 'selected' : ''}>Off</option>
                        <option value="true" ${el.batch_process_mode ? 'selected' : ''}>On</option>
                    </select>
                </div>
                ${el.batch_process_mode ? `
                <div class="prop-row"><span class="prop-label">Accumulate</span><input type="number" class="prop-input" value="${el.batch_process_size || 10}" onchange="updateProp('batch_process_size', +this.value)" min="2" title="Parts to collect before firing"><span class="prop-unit">pcs</span></div>
                <div class="prop-row"><span class="prop-label">Cycle (batch)</span><input type="number" class="prop-input" value="${el.batch_process_time || 300}" onchange="updateProp('batch_process_time', +this.value)" min="1" title="Fixed processing time for the whole batch"><span class="prop-unit">s</span></div>
                <div style="font-size:0.5rem;color:var(--text-dim);line-height:1.3;margin-top:2px;">Accumulation mode (oven/furnace). Waits for N parts, processes all at once. Creates WIP bulges before and starvation after.</div>
                ` : ''}
                <div class="prop-row">
                    <span class="prop-label">Assembly</span>
                    <select class="prop-input" onchange="updateProp('assembly_mode', this.value === 'true'); showProperties(selectedElement, 'machine');">
                        <option value="false" ${!el.assembly_mode ? 'selected' : ''}>Off</option>
                        <option value="true" ${el.assembly_mode ? 'selected' : ''}>On</option>
                    </select>
                </div>
                ${el.assembly_mode ? (() => {
                    const allPTs = getAllProductTypes();
                    const inputs = el.assembly_inputs || {};
                    const inputRows = allPTs.map(pt => `
                        <div class="prop-row">
                            <span class="prop-label" style="width:40px;">${pt}</span>
                            <input type="number" class="prop-input" value="${inputs[pt] || 0}"
                                onchange="updateAssemblyInput('${pt}', +this.value)"
                                min="0" style="width:40px;" title="Number of ${pt} parts needed per assembly">
                            <span class="prop-unit">ea</span>
                        </div>
                    `).join('');
                    return allPTs.length > 0 ? `<div style="font-size:0.5rem;color:var(--text-dim);margin-bottom:2px;">Parts needed per assembly:</div>${inputRows}`
                        : '<div style="font-size:0.5rem;color:var(--text-dim);">Add product types on a Source first</div>';
                })() : '<div style="font-size:0.5rem;color:var(--text-dim);">Enable to require multiple input parts</div>'}
            </div>
        `;
    } else if (type === 'source') {
        const pts = el.product_types || [{ name: 'A', ratio: 1.0, color: '#4a9f6e' }];
        const ptRows = pts.map((pt, i) => `
            <div class="product-type-row">
                <input type="color" class="pt-color" value="${pt.color || '#4a9f6e'}" onchange="updateProductType(${i}, 'color', this.value)" style="width:14px;height:14px;padding:0;border:none;cursor:pointer;">
                <input class="prop-input" value="${pt.name}" onchange="updateProductType(${i}, 'name', this.value)" style="width:40px;" placeholder="Name">
                <input type="number" class="prop-input" value="${pt.ratio}" onchange="updateProductType(${i}, 'ratio', +this.value)" style="width:40px;" step="0.1" min="0" placeholder="Ratio">
                <input type="number" class="prop-input" value="${pt.shelf_life || ''}" onchange="updateProductType(${i}, 'shelf_life', +this.value || 0)" style="width:40px;" placeholder="∞" min="0" title="Shelf life in seconds. 0 = never expires.">
                <button class="pt-remove" onclick="removeProductType(${i})">×</button>
            </div>
        `).join('');

        content.innerHTML = `
            <div class="prop-row"><span class="prop-label">Name</span><input class="prop-input" value="${el.name || ''}" onchange="updateProp('name', this.value)"></div>
            <div class="prop-row"><span class="prop-label">Arrival Rate</span><input type="number" class="prop-input" value="${el.arrival_rate || 60}" onchange="updateProp('arrival_rate', +this.value)"><span class="prop-unit">s</span></div>
            <div class="prop-row"><span class="prop-label">Distribution</span>
                <select class="prop-input" onchange="updateProp('arrival_distribution', this.value)">
                    <option value="exponential" ${el.arrival_distribution === 'exponential' ? 'selected' : ''}>Exponential</option>
                    <option value="constant" ${el.arrival_distribution === 'constant' ? 'selected' : ''}>Constant</option>
                    <option value="normal" ${el.arrival_distribution === 'normal' ? 'selected' : ''}>Normal</option>
                </select>
            </div>
            <div style="margin-top:8px;">
                <span class="prop-label" style="width:auto; display:block; margin-bottom:4px;">Product Mix</span>
                ${ptRows}
                <button class="sim-btn" onclick="addProductType()" style="font-size:0.65rem; padding:3px 8px; margin-top:2px;">+ Product</button>
            </div>
            <div class="prop-row" style="margin-top:8px;">
                <span class="prop-label">Schedule</span>
                <select class="prop-input" onchange="updateProp('schedule_mode', this.value)">
                    <option value="fixed_mix" ${(el.schedule_mode || 'fixed_mix') === 'fixed_mix' ? 'selected' : ''}>Fixed Mix (push)</option>
                    <option value="kanban" ${el.schedule_mode === 'kanban' ? 'selected' : ''}>Kanban (pull)</option>
                    <option value="chase_demand" ${el.schedule_mode === 'chase_demand' ? 'selected' : ''}>Chase Demand</option>
                    <option value="batch_sequence" ${el.schedule_mode === 'batch_sequence' ? 'selected' : ''}>Batch Sequence</option>
                </select>
            </div>
            <div style="font-size:0.55rem; color:var(--text-dim); margin-top:2px; line-height:1.3;">
                ${(el.schedule_mode || 'fixed_mix') === 'fixed_mix' ? 'Produces per mix ratios regardless of FG levels' :
                  el.schedule_mode === 'kanban' ? 'Produces whichever product is below reorder point on FG sink' :
                  el.schedule_mode === 'chase_demand' ? 'Shifts mix toward products farthest below safety stock' :
                  'Runs full batches of one product, then switches to lowest FG'}
            </div>
            <div style="margin-top:8px; border-top:1px solid var(--border); padding-top:6px;">
                <span class="prop-label" style="width:auto; display:block; margin-bottom:4px;">Supplier Reliability</span>
                <div class="prop-row"><span class="prop-label">On-Time %</span><input type="number" class="prop-input" value="${((el.supplier_reliability ?? 1) * 100).toFixed(0)}" onchange="updateProp('supplier_reliability', +this.value/100)" step="5" min="0" max="100" title="Probability each delivery arrives on time"><span class="prop-unit">%</span></div>
                <div class="prop-row"><span class="prop-label">Late Delay</span><input type="number" class="prop-input" value="${el.late_delivery_penalty || ''}" onchange="updateProp('late_delivery_penalty', +this.value || 0)" placeholder="auto" min="0" title="Extra seconds when supplier is late. 0 = random 0.5-2× interarrival."><span class="prop-unit">s</span></div>
                <div class="prop-row"><span class="prop-label">Mat'l Qual</span><input type="number" class="prop-input" value="${((el.incoming_quality_rate ?? 1) * 100).toFixed(0)}" onchange="updateProp('incoming_quality_rate', +this.value/100)" step="5" min="0" max="100" title="Incoming material quality. Bad batches slow processing and increase scrap."><span class="prop-unit">%</span></div>
                <div class="prop-row"><span class="prop-label">Mat'l Cost</span><input type="number" class="prop-input" value="${el.material_cost_per_unit || ''}" onchange="updateProp('material_cost_per_unit', +this.value || 0)" placeholder="0" min="0" title="Material cost per unit for cost accounting"><span class="prop-unit">$/unit</span></div>
            </div>
            <div style="margin-top:8px; border-top:1px solid var(--border); padding-top:6px;">
                <span class="prop-label" style="width:auto; display:block; margin-bottom:4px;">Rush Orders & Due Dates</span>
                <div class="prop-row"><span class="prop-label">Rush %</span><input type="number" class="prop-input" value="${((el.rush_order_rate || 0) * 100).toFixed(0)}" onchange="updateProp('rush_order_rate', +this.value/100)" step="1" min="0" max="50"><span class="prop-unit">%</span></div>
                <div class="prop-row"><span class="prop-label">Due Date</span><input type="number" class="prop-input" value="${el.due_date_target || ''}" onchange="updateProp('due_date_target', +this.value || 0)" placeholder="disabled" min="0"><span class="prop-unit">s</span></div>
                <div class="prop-row"><span class="prop-label">Late @</span><input type="number" class="prop-input" value="${((el.lateness_threshold || 0.8) * 100).toFixed(0)}" onchange="updateProp('lateness_threshold', +this.value/100)" step="5" min="50" max="100"><span class="prop-unit">%</span></div>
                <div style="font-size:0.5rem; color:var(--text-dim); line-height:1.3; margin-top:2px;">
                    Rush orders arrive with 50% tighter due dates and jump queues. Normal orders are promoted to rush when they exceed the lateness threshold — triggering extra changeovers that cascade delays.
                </div>
            </div>
        `;
    } else if (type === 'sink') {
        const sinkMode = el.sink_mode || 'exit';
        const allPTs = getAllProductTypes();
        const demandRates = el.demand_rates || [];
        const safetyStock = el.safety_stock || {};
        const reorderPoint = el.reorder_point || {};

        // Build demand rows for each product type
        const demandRowsHTML = allPTs.map(pt => {
            const dr = demandRates.find(d => d.product === pt) || {};
            const ss = safetyStock[pt] ?? '';
            const rp = reorderPoint[pt] ?? '';
            return `
                <div style="font-size:0.65rem; color:var(--text-dim); margin-top:6px; font-weight:600;">Product ${pt}</div>
                <div class="prop-row"><span class="prop-label" style="width:55px;">Demand</span><input type="number" class="prop-input" value="${dr.rate || ''}" onchange="updateDemandRate('${pt}', 'rate', +this.value)" placeholder="0" min="0"><span class="prop-unit">/hr</span></div>
                <div class="prop-row"><span class="prop-label" style="width:55px;">Safety Stk</span><input type="number" class="prop-input" value="${ss}" onchange="updateSafetyStock('${pt}', +this.value)" placeholder="0" min="0"></div>
                <div class="prop-row"><span class="prop-label" style="width:55px;">Reorder Pt</span><input type="number" class="prop-input" value="${rp}" onchange="updateReorderPoint('${pt}', +this.value)" placeholder="0" min="0"></div>
            `;
        }).join('');

        const noPTs = allPTs.length === 0 ? '<div style="font-size:0.6rem; color:var(--text-dim); margin-top:4px;">Add product types on a Source first</div>' : '';

        content.innerHTML = `
            <div class="prop-row"><span class="prop-label">Name</span><input class="prop-input" value="${el.name || ''}" onchange="updateProp('name', this.value)"></div>
            <div class="prop-row"><span class="prop-label">Mode</span>
                <select class="prop-input" onchange="updateProp('sink_mode', this.value); showProperties(selectedElement, 'sink');">
                    <option value="exit" ${sinkMode === 'exit' ? 'selected' : ''}>Exit (parts vanish)</option>
                    <option value="fg_warehouse" ${sinkMode === 'fg_warehouse' ? 'selected' : ''}>FG Warehouse</option>
                </select>
            </div>
            ${sinkMode === 'fg_warehouse' ? `
                <div style="margin-top:6px; border-top: 1px solid var(--border); padding-top:6px;">
                    <span class="prop-label" style="width:auto; display:block; margin-bottom:4px;">Customer Demand & Stocking</span>
                    ${noPTs}${demandRowsHTML}
                </div>
            ` : ''}
        `;
    } else if (type === 'connection') {
        const fromEl = findElement(el.from_id);
        const toEl = findElement(el.to_id);
        const ttOpts = Object.entries(TRANSPORT_TYPES).map(([k, v]) =>
            `<option value="${k}" ${(el.transport_type || 'none') === k ? 'selected' : ''}>${v.label}</option>`
        ).join('');
        const tt = TRANSPORT_TYPES[el.transport_type || 'none'];
        const estTime = tt.speed !== Infinity && el.transport_distance > 0
            ? `~${(el.transport_distance / tt.speed).toFixed(0)}s`
            : '—';

        content.innerHTML = `
            <div style="font-size:0.72rem; color:var(--text-dim); margin-bottom:8px;">${fromEl?.name || el.from_id} → ${toEl?.name || el.to_id}</div>
            <div class="prop-row"><span class="prop-label">Buffer Cap</span><input type="number" class="prop-input" value="${el.buffer_capacity ?? ''}" onchange="updateProp('buffer_capacity', this.value === '' ? null : +this.value)" placeholder="∞ (infinite)" min="1"></div>
            <div class="prop-row"><span class="prop-label">Transport</span>
                <select class="prop-input" onchange="updateProp('transport_type', this.value); showProperties(selectedElement, 'connection');">
                    ${ttOpts}
                </select>
            </div>
            <div class="prop-row"><span class="prop-label">Distance</span><input type="number" class="prop-input" value="${el.transport_distance || 0}" onchange="updateProp('transport_distance', +this.value)" min="0"><span class="prop-unit">m</span></div>
            <div style="font-size:0.6rem; color:var(--text-dim); margin-top:2px;">
                Est. move: ${estTime}
                ${tt.needsOperator ? ' · needs operator' : ''}
                ${tt.breakdownRate > 0 ? ` · ${(tt.breakdownRate * 100).toFixed(1)}% breakdown risk` : ''}
                ${tt.maxContainerKg ? ` · max ${tt.maxContainerKg}kg` : ''}
            </div>
            <div style="margin-top:6px; border-top:1px solid var(--border); padding-top:6px;">
                <span class="prop-label" style="width:auto; display:block; margin-bottom:2px; font-size:0.6rem;">Product Routing <span style="color:var(--text-dim)">(empty = all products)</span></span>
                <div id="routing-rules-list">
                    ${(el.routing_rules || []).map((r, i) => `
                        <div style="display:flex; gap:4px; align-items:center; margin-bottom:3px;">
                            <select class="prop-input" style="flex:1; font-size:0.55rem; padding:1px 2px;"
                                onchange="updateRoutingRule(${i}, 'product', this.value)">
                                ${getAllProductTypes().map(pt =>
                                    `<option value="${pt}" ${r.product === pt ? 'selected' : ''}>${pt}</option>`
                                ).join('')}
                            </select>
                            <input type="number" class="prop-input" value="${((r.weight ?? 1) * 100).toFixed(0)}"
                                onchange="updateRoutingRule(${i}, 'weight', +this.value/100)"
                                min="1" max="100" step="5" style="width:40px; font-size:0.55rem; padding:1px 2px;"
                                title="Routing weight (100% = always, 50% = half the time)">
                            <span style="font-size:0.5rem;color:var(--text-dim);">%</span>
                            <button onclick="removeRoutingRule(${i})" style="background:none;border:none;color:var(--text-dim);cursor:pointer;font-size:0.7rem;padding:0 2px;" title="Remove rule">✕</button>
                        </div>
                    `).join('')}
                </div>
                <button onclick="addRoutingRule()" style="font-size:0.55rem; padding:2px 6px; background:var(--bg-tertiary); border:1px solid var(--border); color:var(--text-primary); cursor:pointer; border-radius:3px; margin-top:2px;">+ Route Rule</button>
            </div>
        `;
    }
}

function updateProp(key, value) {
    if (!selectedElement) return;
    selectedElement[key] = value;
    renderCanvas();
}

// Product routing rules for connections (CR-4)
function addRoutingRule() {
    if (!selectedElement) return;
    if (!selectedElement.routing_rules) selectedElement.routing_rules = [];
    const pts = getAllProductTypes();
    selectedElement.routing_rules.push({ product: pts[0] || 'Product', weight: 1.0 });
    renderCanvas();
    showProperties(selectedElement, 'connection');
}
function updateRoutingRule(idx, field, value) {
    if (!selectedElement || !selectedElement.routing_rules) return;
    selectedElement.routing_rules[idx][field] = value;
    renderCanvas();
}
function removeRoutingRule(idx) {
    if (!selectedElement || !selectedElement.routing_rules) return;
    selectedElement.routing_rules.splice(idx, 1);
    renderCanvas();
    showProperties(selectedElement, 'connection');
}

// Per-product quality rate overrides (CR-3)
function updateQualityByProduct(productType, field, valueStr) {
    if (!selectedElement) return;
    const qbp = selectedElement.quality_by_product || {};
    if (!qbp[productType]) qbp[productType] = {};
    if (valueStr === '' || valueStr == null) {
        delete qbp[productType][field];
        if (Object.keys(qbp[productType]).length === 0) delete qbp[productType];
    } else {
        qbp[productType][field] = +valueStr / 100;
    }
    selectedElement.quality_by_product = qbp;
    renderCanvas();
}

// Assembly input management
function updateAssemblyInput(productType, qty) {
    if (!selectedElement) return;
    if (!selectedElement.assembly_inputs) selectedElement.assembly_inputs = {};
    if (qty > 0) {
        selectedElement.assembly_inputs[productType] = qty;
    } else {
        delete selectedElement.assembly_inputs[productType];
    }
    renderCanvas();
}

// Product mix management (source properties)
const PT_COLORS = ['#4a9f6e', '#3b82f6', '#f59e0b', '#e74c3c', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];

function updateProductType(idx, field, value) {
    if (!selectedElement || !selectedElement.product_types) return;
    selectedElement.product_types[idx][field] = value;
    showProperties(selectedElement, 'source');
    renderCanvas();
}

function addProductType() {
    if (!selectedElement) return;
    if (!selectedElement.product_types) selectedElement.product_types = [];
    const n = selectedElement.product_types.length;
    const name = String.fromCharCode(65 + n); // A, B, C, D...
    selectedElement.product_types.push({ name, ratio: 0.5, color: PT_COLORS[n % PT_COLORS.length] });
    showProperties(selectedElement, 'source');
    renderCanvas();
}

function removeProductType(idx) {
    if (!selectedElement || !selectedElement.product_types) return;
    if (selectedElement.product_types.length <= 1) { showToast('Need at least one product type'); return; }
    selectedElement.product_types.splice(idx, 1);
    showProperties(selectedElement, 'source');
    renderCanvas();
}

// Setup matrix management (machine properties)
function getAllProductTypes() {
    const types = new Set();
    for (const src of layout.sources) {
        for (const pt of (src.product_types || [])) {
            types.add(pt.name);
        }
    }
    return Array.from(types).sort();
}

function updateSetupMatrix(fromType, toType, value) {
    if (!selectedElement) return;
    if (!selectedElement.setup_matrix) selectedElement.setup_matrix = {};
    const key = `${fromType}→${toType}`;
    if (value === '' || value == null) {
        delete selectedElement.setup_matrix[key];
    } else {
        selectedElement.setup_matrix[key] = +value;
    }
}

// Sink demand/stocking management
function updateDemandRate(product, field, value) {
    if (!selectedElement) return;
    if (!selectedElement.demand_rates) selectedElement.demand_rates = [];
    let dr = selectedElement.demand_rates.find(d => d.product === product);
    if (!dr) {
        dr = { product, rate: 0, distribution: 'exponential', cv: 0 };
        selectedElement.demand_rates.push(dr);
    }
    dr[field] = value;
}

function updateSafetyStock(product, value) {
    if (!selectedElement) return;
    if (!selectedElement.safety_stock) selectedElement.safety_stock = {};
    selectedElement.safety_stock[product] = value;
}

function updateReorderPoint(product, value) {
    if (!selectedElement) return;
    if (!selectedElement.reorder_point) selectedElement.reorder_point = {};
    selectedElement.reorder_point[product] = value;
}

function renderSetupMatrixHTML(el) {
    const types = getAllProductTypes();
    if (types.length < 2) return '<div style="font-size:0.6rem; color:var(--text-dim); margin-top:4px;">Add 2+ product types on a Source to configure setup matrix</div>';

    const matrix = el.setup_matrix || {};
    const defaultCO = el.changeover_time || 0;

    let html = '<table class="setup-matrix-table"><tr><th>From \\ To</th>';
    for (const t of types) html += `<th>${t}</th>`;
    html += '</tr>';
    for (const from of types) {
        html += `<tr><th>${from}</th>`;
        for (const to of types) {
            if (from === to) {
                html += '<td style="background:var(--bg-primary);">—</td>';
            } else {
                const key = `${from}→${to}`;
                const val = matrix[key] ?? '';
                html += `<td><input type="number" value="${val}" placeholder="${defaultCO}" min="0" onchange="updateSetupMatrix('${from}','${to}',this.value)"></td>`;
            }
        }
        html += '</tr>';
    }
    html += '</table>';
    return html;
}

function hideProperties() {
    document.getElementById('props-panel').classList.remove('active');
    selectedElement = null;
}

function deleteSelected() {
    if (!selectedElement) return;
    const id = selectedElement.id;
    layout.sources = layout.sources.filter(s => s.id !== id);
    layout.stations = layout.stations.filter(s => s.id !== id);
    layout.sinks = layout.sinks.filter(s => s.id !== id);
    // Remove the connection itself, or connections attached to a deleted element
    layout.connections = layout.connections.filter(c => c.id !== id && c.from_id !== id && c.to_id !== id);
    hideProperties();
    renderCanvas();
}

// Drag elements on canvas
function startDrag(e, el) {
    if (currentTool !== 'select') return;
    e.stopPropagation();
    dragElement = el;
    const svg = document.getElementById('sim-canvas');
    const pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const svgP = pt.matrixTransform(document.getElementById('transform-group').getScreenCTM().inverse());
    dragOffset.x = svgP.x - el.x;
    dragOffset.y = svgP.y - el.y;
}

// Canvas events
const canvas = document.getElementById('sim-canvas');

canvas.addEventListener('mousemove', (e) => {
    if (dragElement) {
        const svg = document.getElementById('sim-canvas');
        const pt = svg.createSVGPoint();
        pt.x = e.clientX; pt.y = e.clientY;
        const svgP = pt.matrixTransform(document.getElementById('transform-group').getScreenCTM().inverse());
        dragElement.x = Math.round((svgP.x - dragOffset.x) / 10) * 10;
        dragElement.y = Math.round((svgP.y - dragOffset.y) / 10) * 10;
        renderCanvas();
    }
    if (isPanning) {
        canvasPanX += e.movementX;
        canvasPanY += e.movementY;
        updateTransform();
    }
});

canvas.addEventListener('mouseup', () => {
    dragElement = null;
    isPanning = false;
});

const canvasWrap = document.getElementById('canvas-wrap');
canvasWrap.addEventListener('mousedown', (e) => {
    // Place element if in placing mode
    if (placingType) {
        placeElement(e);
        return;
    }
    if (currentTool === 'pan') {
        isPanning = true;
    }
    if (currentTool === 'select' && (e.target === canvas || e.target === canvasWrap || e.target.tagName === 'rect')) {
        hideProperties();
        renderCanvas();
    }
});

canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    canvasZoom = Math.max(0.2, Math.min(3, canvasZoom * factor));
    updateTransform();
});

// Click-to-place from palette
let placingType = null;

document.querySelectorAll('.palette-item').forEach(item => {
    item.addEventListener('click', () => {
        const type = item.dataset.type;
        if (type === 'buffer') {
            showToast('Connect two elements and set buffer capacity on the connection');
            return;
        }
        // Toggle placing mode
        if (placingType === type) {
            placingType = null;
            document.querySelectorAll('.palette-item').forEach(p => p.classList.remove('placing'));
            document.getElementById('sim-canvas').style.cursor = '';
            return;
        }
        placingType = type;
        document.querySelectorAll('.palette-item').forEach(p => p.classList.remove('placing'));
        item.classList.add('placing');
        document.getElementById('sim-canvas').style.cursor = 'crosshair';
        showToast('Click on canvas to place ' + (type === 'workcenter' ? 'Work Center' : type.charAt(0).toUpperCase() + type.slice(1)));
    });
});

function placeElement(e) {
    if (!placingType) return;

    const svg = document.getElementById('sim-canvas');
    const rect = svg.getBoundingClientRect();
    const rawX = (e.clientX - rect.left - canvasPanX) / canvasZoom;
    const rawY = (e.clientY - rect.top - canvasPanY) / canvasZoom;
    const x = Math.round(rawX / 10) * 10;
    const y = Math.round(rawY / 10) * 10;

    if (placingType === 'source') {
        layout.sources.push({ id: genId('src'), name: 'Source', x, y, arrival_distribution: 'exponential', arrival_rate: 60, arrival_cv: 0, batch_size: 1, product_types: [{ name: 'A', ratio: 1.0, color: '#4a9f6e' }], schedule_mode: 'fixed_mix' });
    } else if (placingType === 'machine') {
        layout.stations.push({ id: genId('stn'), type: 'single', name: `Stn ${layout.stations.length + 1}`, x, y, cycle_time: 30, cycle_time_cv: 0.15, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 1, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} });
    } else if (placingType === 'sink') {
        layout.sinks.push({ id: genId('sink'), name: 'FG Warehouse', x, y, sink_mode: 'fg_warehouse', demand_rates: [], safety_stock: {}, reorder_point: {} });
    } else if (placingType === 'workcenter') {
        layout.work_centers.push({ id: genId('wc'), name: `WC ${layout.work_centers.length + 1}`, x, y, width: 220, height: 160 });
    }

    // Exit placing mode after placement
    placingType = null;
    document.querySelectorAll('.palette-item').forEach(p => p.classList.remove('placing'));
    svg.style.cursor = '';
    renderCanvas();
}

// =============================================================================
// Tools
// =============================================================================

function setTool(tool) {
    currentTool = tool;
    document.querySelectorAll('.sv-btn-icon').forEach(b => b.classList.remove('active'));
    document.getElementById(`tool-${tool}`).classList.add('active');
    const cvs = document.getElementById('sim-canvas');
    cvs.className = `sim-canvas${tool === 'pan' ? ' pan-mode' : tool === 'connect' ? ' connect-mode' : ''}`;
    connectFrom = null;
}

function zoomCanvas(factor) {
    canvasZoom = Math.max(0.2, Math.min(3, canvasZoom * factor));
    updateTransform();
}

function resetView() {
    canvasZoom = 1;
    canvasPanX = 50;
    canvasPanY = 50;
    updateTransform();
}
