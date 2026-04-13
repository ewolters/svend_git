/**
 * calc-core.js — Core infrastructure for Operations Workbench
 *
 * Load order: after svend-math.js, svend-sim-core.js
 * Extracted from: calculators.html (inline script)
 *
 * Provides: SvendOps data bus, calcMeta registry, calcGuide content,
 * navigation (showCalc, filterCalcs, navigateToCalc), stepper widget,
 * MonteCarlo engine, renderNextSteps, guide system.
 */

// ============================================================================
// Navigation
// ============================================================================

const calcMeta = {
    takt: { title: 'Takt Time', desc: 'Calculate the pace of production to meet customer demand' },
    rto: { title: 'RTO (Required to Operate)', desc: 'Determine staffing requirements with variation buffer' },
    yamazumi: { title: 'Yamazumi Chart', desc: 'Visualize line balance and identify imbalances' },
    'line-sim': { title: 'Line Simulator', desc: 'Watch WIP flow through stations in real-time. See bottlenecks form. Run line balancing events.' },
    kanban: { title: 'Kanban Sizing', desc: 'Calculate the number of kanban cards needed' },
    epei: { title: 'EPEI (Every Part Every Interval)', desc: 'Determine production interval for part mix' },
    safety: { title: 'Safety Stock', desc: 'Calculate buffer inventory for demand and lead time variation' },
    eoq: { title: 'Economic Order Quantity', desc: 'Optimize order size to minimize total inventory cost' },
    oee: { title: 'OEE (Overall Equipment Effectiveness)', desc: 'Measure equipment productivity: Availability × Performance × Quality' },
    bottleneck: { title: 'Bottleneck Analysis', desc: 'Identify the constraint limiting system throughput' },
    littles: { title: "Little's Law", desc: 'WIP = Throughput × Lead Time — the fundamental flow equation' },
    queue: { title: 'M/M/c Queue', desc: 'Classic multi-server queue with exponential arrivals and service — call centers, checkout lanes, help desks' },
    'queue-finite': { title: 'M/M/c/K Finite Queue', desc: 'Limited queue capacity — customers balk when queue is full (drive-throughs, ERs with diversion)' },
    'queue-priority': { title: 'Priority Queue', desc: 'Multiple priority classes with preemption — ER triage, tiered support, VIP lanes' },
    'queue-optimizer': { title: 'Staffing Optimizer', desc: 'Find optimal server count given costs and service level targets' },
    'queue-sim': { title: 'Live Queue Simulator', desc: 'Watch queues form in real-time with Monte Carlo variability' },
    'queue-compare': { title: 'A/B Scenario Compare', desc: 'Run two simulations side-by-side — current vs proposed, see the difference live' },
    'queue-tandem': { title: 'Multi-Stage Queue', desc: 'Tandem queues — patient flows through triage → doctor → checkout, manufacturing lines' },
    'erlang': { title: 'Erlang C Staffing', desc: 'Calculate optimal staffing for call centers, ERs, help desks, and service operations' },
    pitch: { title: 'Pitch', desc: 'Takt × pack-out quantity for paced material withdrawal' },
    pfa: { title: 'Product Flow Analysis', desc: 'TIPS observation — follow the product through Transport, Inspect, Process, Storage' },
    wfa: { title: 'Workflow Analysis', desc: 'Therblig-based task breakdown — VA, required NVA, and unnecessary waste' },
    rty: { title: 'Rolled Throughput Yield', desc: 'Calculate first-pass yield across multiple process steps' },
    dpmo: { title: 'DPMO & Sigma Level', desc: 'Convert defect rates to sigma levels and vice versa' },
    turns: { title: 'Inventory Turns', desc: 'Calculate inventory turnover rate and days on hand' },
    coq: { title: 'Cost of Quality', desc: 'Analyze prevention, appraisal, and failure costs' },
    smed: { title: 'SMED Analysis', desc: 'Single Minute Exchange of Die — classify and reduce changeover elements' },
    changeover: { title: 'Changeover Matrix', desc: 'Sequence-dependent setup times between products' },
    fmea: { title: 'FMEA / RPN', desc: 'Failure Mode & Effects Analysis — Severity × Occurrence × Detection' },
    cpk: { title: 'Cp / Cpk Calculator', desc: 'Quick process capability check against spec limits' },
    samplesize: { title: 'Sample Size', desc: 'Calculate required sample size for studies and audits' },
    poweranalysis: { title: 'Power Analysis', desc: 'Interactive sample size calculator with power curves — t-tests, proportions, ANOVA' },
    'riskmatrix': { title: 'Risk Matrix', desc: '5x5 likelihood x severity risk assessment heat map — projects, safety, IT, compliance' },
    lineeff: { title: 'Line Efficiency', desc: 'Theoretical vs actual rate, losses by category' },
    ole: { title: 'OLE (Overall Labor Effectiveness)', desc: 'Availability × Performance × Quality for labor' },
    cycletime: { title: 'Cycle Time Study', desc: 'VA / NVA / Wait breakdown with visualization' },
    'mtbf': { title: 'MTBF / MTTR + Availability', desc: 'Mean time between failures, mean time to repair, and system availability with uptime nines' },
    beforeafter: { title: 'Before / After', desc: 'Compare metrics pre and post improvement' },
    heijunka: { title: 'Heijunka (Leveling)', desc: 'Production mix smoothing and sequencing' },
    // Scheduling
    sequencer: { title: 'Job Sequencer', desc: 'Drag-and-drop job scheduling with live metrics — makespan, tardiness, setup time' },
    'seq-optimizer': { title: 'Sequence Optimizer', desc: 'Find optimal job sequence to minimize setup time or tardiness' },
    'capacity-load': { title: 'Capacity Load Chart', desc: 'Visualize work load against available capacity by time period' },
    'mixed-model': { title: 'Mixed-Model Sequencer', desc: 'Level production mix to smooth workflow — Toyota\'s heijunka sequencing' },
    'due-date-sim': { title: 'Due Date Risk Simulator', desc: 'Monte Carlo simulation of on-time delivery probability' },
    // Method
    qfd: { title: 'House of Quality (QFD)', desc: 'Quality Function Deployment — cascade customer requirements through design, parts, process, and production' },
    // Simulators
    'kanban-sim': { title: 'Kanban Pull System Simulator', desc: 'Watch kanban cards circulate. Toggle PUSH vs PULL and see why pull systems control WIP.' },
    'beer-game': { title: 'Beer Game — Bullwhip Effect', desc: 'The MIT Beer Game. Watch small demand changes amplify through a 4-tier supply chain.' },
    'toc-sim': { title: 'TOC / Drum-Buffer-Rope', desc: 'Theory of Constraints in action. Protect the bottleneck, control the system.' },
    'safety-sim': { title: 'Safety Stock Simulator', desc: 'Simulate inventory over 180+ days with stochastic demand. Watch stockouts happen — and see your safety stock formula in action.' },
    'heijunka-sim': { title: 'Heijunka Simulator', desc: 'Batched vs leveled production side-by-side. See the WIP and lead time difference.' },
    'smed-sim': { title: 'SMED Simulator', desc: 'Animate changeover reduction. Drag elements from internal to external and watch the bar shrink.' },
    'cell-sim': { title: 'Cell Design Simulator', desc: 'Watch spaghetti diagrams form as operators walk. Compare straight-line, U-cell, L-cell, and parallel layouts.' },
    'fmea-sim': { title: 'FMEA Monte Carlo', desc: 'Run failure cascades. See how compounding small risks create system-level failures.' },
    // Quality & DOE
    'desirability': { title: 'Multi-Response Desirability', desc: 'Optimize multiple responses simultaneously. Drag sliders to find the sweet spot.' },
    'spc-rare': { title: 'SPC Rare Events Lab', desc: 'G chart and T chart for rare events. Generate data, inject shifts, watch detection live.' },
    'probit': { title: 'Probit / Dose-Response', desc: 'Fit probit or logit models to dose-response data. Get ED50/LD50 with confidence intervals.' },
    // Process Belief System
    'pbs-cpk': { title: 'Bayesian Cpk', desc: 'Posterior distribution of process capability. Get credible intervals and P(Cpk > 1.33) from your data.' },
    'pbs-belief': { title: 'Belief Chart (BOCPD)', desc: 'Bayesian Online Changepoint Detection. Find where your process shifted — no window tuning needed.' },
    'pbs-evidence': { title: 'Evidence Strength', desc: 'E-value accumulation for sequential evidence against a reference mean. Anytime-valid inference.' },
    'pbs-sigma': { title: 'Bayesian Sigma', desc: 'Escape probability via quasipotential landscape. How far is the process from spec limits, accounting for momentum and uncertainty?' },
    // SIOP
    'abc': { title: 'ABC Analysis', desc: 'Pareto classification of items by cumulative value — A (vital few), B (moderate), C (trivial many)' },
    'demand-profile': { title: 'Demand Profile', desc: 'Syntetos-Boylan demand pattern classification — smooth, erratic, intermittent, or lumpy' },
    'service-level': { title: 'Service Level Trade-off', desc: 'Optimal fill rate balancing holding cost vs stockout cost' },
    'mrp': { title: 'MRP Netting', desc: 'Gross-to-net requirements explosion with planned order releases' },
};

// ============================================================================
// Shared State — Cross-Calculator Data Bus
// ============================================================================

const SvendOps = {
    // Published values from calculators
    values: {},

    // Publish a value (other calculators can pull from this)
    publish(key, value, unit, source) {
        this.values[key] = { value, unit, source, timestamp: Date.now() };
        this.updateLinkIndicators();
    },

    // Get a published value
    get(key) {
        return this.values[key]?.value || null;
    },

    // Update visual indicators showing available linked values
    updateLinkIndicators() {
        document.querySelectorAll('[data-link-from]').forEach(el => {
            const key = el.dataset.linkFrom;
            if (this.values[key]) {
                el.classList.add('has-link');
                el.title = `Pull from ${this.values[key].source}: ${this.values[key].value} ${this.values[key].unit}`;
            }
        });
    },

    // Pull a value into an input field with visual feedback
    pull(key, targetId) {
        const val = this.get(key);
        if (val !== null) {
            const el = document.getElementById(targetId);
            el.value = val;
            el.dispatchEvent(new Event('input'));
            // Flash green highlight on target input
            el.style.transition = 'box-shadow 0.2s';
            el.style.boxShadow = '0 0 0 2px rgba(74, 159, 110, 0.5)';
            setTimeout(() => { el.style.boxShadow = ''; }, 1200);
        } else {
            showToast('No data available yet — run the source calculator first');
        }
    }
};

// ============================================================================
// Guide Content — Semi-Smart Calculator Help
// ============================================================================

const calcGuide = {
    takt: {
        purpose: 'Calculates the pace of production needed to meet customer demand. The heartbeat of lean — use this when setting up a new line, rebalancing after demand changes, or establishing pitch intervals.',
        inputs: {
            'Available Time': 'Total shift time before subtracting breaks (minutes).',
            'Planned Breaks': 'Lunch, rest periods, planned maintenance (minutes).',
            'Customer Demand': 'Units the customer needs per shift.'
        },
        formula: 'Takt = (Available Time − Breaks) ÷ Demand',
        feedsFrom: [],
        feedsInto: [
            { calcId: 'rto', label: 'RTO Staffing', pullKey: 'takt', pullTarget: 'rto-takt' },
            { calcId: 'yamazumi', label: 'Yamazumi Chart', pullKey: 'takt', pullTarget: 'yama-takt' },
            { calcId: 'pitch', label: 'Pitch', pullKey: 'takt', pullTarget: 'pitch-takt' },
            { calcId: 'line-sim', label: 'Line Simulator', pullKey: 'takt', pullTarget: 'ls-takt' }
        ],
        publishes: ['takt', 'taktMin']
    },
    rto: {
        purpose: 'Determines how many operators a line needs, accounting for cycle time variation. Use after calculating takt time to staff your production line.',
        inputs: {
            'Total Cycle Time': 'Sum of all task times at a station (seconds).',
            'Takt Time': 'Required pace from demand (seconds). Pull from Takt calculator.',
            'Coefficient of Variation': 'Variation buffer — typical values are 5-15%.'
        },
        formula: 'RTO = (Cycle Time ÷ Takt) × (1 + CV%)',
        feedsFrom: [
            { calcId: 'takt', label: 'Takt Time', key: 'takt' }
        ],
        feedsInto: [],
        publishes: ['rtoStaff', 'lineEfficiency']
    },
    oee: {
        purpose: 'Measures equipment productivity across three dimensions: Availability, Performance, Quality. World-class OEE is 85%+. Use to identify which loss category to attack first.',
        inputs: {
            'Planned Production Time': 'Total scheduled production time (minutes).',
            'Actual Run Time': 'Time the equipment was actually running (minutes).',
            'Ideal Cycle Time': 'Fastest possible cycle time per unit. Pull from Bottleneck.',
            'Total Pieces': 'Total units produced during the run.',
            'Good Pieces': 'Units passing quality inspection.'
        },
        formula: 'OEE = Availability × Performance × Quality',
        feedsFrom: [
            { calcId: 'bottleneck', label: 'Bottleneck Analysis', key: 'bottleneckCT' }
        ],
        feedsInto: [
            { calcId: 'bottleneck', label: 'Bottleneck Analysis' },
            { calcId: 'smed', label: 'SMED (reduce changeover)' },
            { calcId: 'lineeff', label: 'Line Efficiency' }
        ],
        publishes: ['oee', 'oeeAvailability']
    },
    bottleneck: {
        purpose: 'Identifies which station constrains system throughput. The bottleneck sets the maximum rate — focus improvement here first (Theory of Constraints).',
        inputs: {
            'Station Names': 'Label for each process step.',
            'Cycle Times': 'Processing time per unit at each station (seconds).'
        },
        formula: 'Throughput = 3600 ÷ max(Cycle Times)',
        feedsFrom: [],
        feedsInto: [
            { calcId: 'oee', label: 'OEE', pullKey: 'bottleneckCT', pullTarget: 'oee-ideal' },
            { calcId: 'littles', label: "Little's Law", pullKey: 'bottleneckThroughput', pullTarget: 'littles-thr' },
            { calcId: 'queue', label: 'M/M/c Queue', pullKey: 'bottleneckThroughput', pullTarget: 'queue-mu' },
            { calcId: 'queue-priority', label: 'Priority Queue', pullKey: 'bottleneckThroughput', pullTarget: 'qp-mu' }
        ],
        publishes: ['bottleneckCT', 'bottleneckThroughput']
    },
    smed: {
        purpose: 'Classify changeover elements as Internal (machine stopped) or External (while running) to reduce downtime. Goal: single-digit minute changeovers.',
        inputs: {
            'Changeover Elements': 'Each task in the changeover process, with time and type classification.'
        },
        formula: 'Internal Time = Σ internal elements (target for reduction)',
        feedsFrom: [],
        feedsInto: [
            { calcId: 'epei', label: 'EPEI', pullKey: 'changeoverInternal', pullTarget: 'epei-changeover' },
            { calcId: 'line-sim', label: 'Line Simulator', pullKey: 'changeoverInternal', pullTarget: 'ls-changeover-time' },
            { calcId: 'lineeff', label: 'Line Efficiency', pullKey: 'changeoverInternal', pullTarget: 'lineeff-changeover' }
        ],
        publishes: ['changeoverInternal']
    },
    epei: {
        purpose: 'Calculates how often you can cycle through all part numbers given changeover constraints. Lower EPEI = more flexible production, smaller batches.',
        inputs: {
            'Available Production Time': 'Hours available per day.',
            'Number of Part Numbers': 'SKUs in the rotation.',
            'Changeover Time': 'Minutes per changeover. Pull from SMED.',
            'C/O Allowance %': 'Percentage of available time allocated to changeovers.'
        },
        formula: 'EPEI = (Parts × C/O Time) ÷ (Available × 60 × Allowance%)',
        feedsFrom: [
            { calcId: 'smed', label: 'SMED', key: 'changeoverInternal' }
        ],
        feedsInto: [
            { calcId: 'heijunka', label: 'Heijunka (Leveling)' }
        ],
        publishes: ['epei']
    },
    kanban: {
        purpose: 'Sizes a pull system. Calculates the number of kanban cards needed to signal replenishment between processes without overproducing.',
        inputs: {
            'Daily Demand': 'Units consumed per day by downstream process.',
            'Lead Time': 'Replenishment lead time in days.',
            'Safety Factor': 'Buffer percentage for demand/supply variation.',
            'Container Size': 'Units per container. Pull from EOQ for optimal size.'
        },
        formula: 'Kanbans = D × LT × (1 + SF) ÷ Container Size',
        feedsFrom: [
            { calcId: 'eoq', label: 'EOQ', key: 'eoq' }
        ],
        feedsInto: [],
        publishes: ['kanbanCards', 'kanbanInventory']
    },
    eoq: {
        purpose: 'Finds the order quantity that minimizes total inventory cost (holding + ordering). Classic operations research result — the square root formula.',
        inputs: {
            'Annual Demand': 'Total units needed per year.',
            'Ordering Cost': 'Cost to place one order ($).',
            'Holding Cost': 'Cost to hold one unit for one year ($).'
        },
        formula: 'EOQ = √(2DS ÷ H)',
        feedsFrom: [],
        feedsInto: [
            { calcId: 'kanban', label: 'Kanban Sizing', pullKey: 'eoq', pullTarget: 'kanban-container' }
        ],
        publishes: ['eoq']
    },
    littles: {
        purpose: "The fundamental law of flow. Connects WIP, throughput, and lead time — if you know any two, you get the third. Applies to any stable system: factory floors, ERs, software pipelines.",
        inputs: {
            'Solve For': 'Choose which variable to calculate.',
            'Two Known Values': 'Enter the two values you know (WIP, throughput, or lead time).'
        },
        formula: 'L = λW  (WIP = Throughput × Lead Time)',
        feedsFrom: [
            { calcId: 'bottleneck', label: 'Bottleneck Analysis', key: 'bottleneckThroughput' }
        ],
        feedsInto: [],
        publishes: ['littlesResult']
    },
    pitch: {
        purpose: 'Sets the rhythm for material withdrawal. Pitch = takt × pack-out quantity, creating a regular heartbeat for the production floor and logistics.',
        inputs: {
            'Takt Time': 'Seconds per unit. Pull from Takt calculator.',
            'Pack-Out Quantity': 'Units per container or pallet.'
        },
        formula: 'Pitch = Takt × Pack-Out Qty',
        feedsFrom: [
            { calcId: 'takt', label: 'Takt Time', key: 'takt' }
        ],
        feedsInto: [
            { calcId: 'heijunka', label: 'Heijunka', pullKey: 'pitch', pullTarget: 'heijunka-pitch' }
        ],
        publishes: ['pitch']
    },
    mtbf: {
        purpose: 'Calculates Mean Time Between Failures and Mean Time to Repair from maintenance records. Availability and "nines" show how close you are to always-on operation.',
        inputs: {
            'Failure Events': 'Enter timestamps of failures and repairs, or enter MTBF/MTTR directly.',
            'Mode': 'Choose between entering raw events or direct values.'
        },
        formula: 'Availability = MTBF ÷ (MTBF + MTTR)',
        feedsFrom: [],
        feedsInto: [
            { calcId: 'oee', label: 'OEE (availability input)' },
            { calcId: 'riskmatrix', label: 'Risk Matrix' }
        ],
        publishes: ['mtbf', 'mttr', 'availability']
    },
    erlang: {
        purpose: 'Calculates optimal staffing for service operations — call centers, ERs, help desks, checkout lines. Uses the Erlang C queuing formula to hit a target service level.',
        inputs: {
            'Arrival Rate': 'Customers/calls arriving per hour.',
            'Avg Service Time': 'Minutes to handle one customer/call.',
            'Target Answer Time': 'Seconds — how fast you want to answer.',
            'Target Service Level': 'Percentage of calls answered within target time.'
        },
        formula: 'Erlang C: P(wait) = [A^c/c! × c/(c−A)] ÷ [Σ A^k/k! + A^c/c! × c/(c−A)]',
        feedsFrom: [],
        feedsInto: [
            { calcId: 'queue-sim', label: 'Queue Simulator' },
            { calcId: 'coq', label: 'Cost of Quality' }
        ],
        publishes: ['erlang_agents', 'erlang_sl']
    },
    riskmatrix: {
        purpose: 'A 5×5 likelihood × severity risk assessment. Add risks, score them, and see the heat map. Works for project risks, safety hazards, IT incidents, compliance gaps.',
        inputs: {
            'Risk Description': 'What could go wrong.',
            'Likelihood': '1 (rare) to 5 (almost certain).',
            'Severity': '1 (negligible) to 5 (catastrophic).'
        },
        formula: 'Risk Score = Likelihood × Severity',
        feedsFrom: [],
        feedsInto: [
            { calcId: 'fmea', label: 'FMEA / RPN (deeper analysis)' },
            { calcId: 'mtbf', label: 'MTBF / MTTR' }
        ],
        publishes: []
    },
    'cell-sim': {
        purpose: 'Simulates operator movement across different manufacturing cell layouts. The spaghetti diagram reveals wasted walking — the #1 argument for U-cells over straight lines. Use when designing or redesigning a production cell.',
        inputs: {
            'Layout Type': 'Straight line, U-cell, L-cell, or parallel — determines station geometry.',
            'Stations': 'Number of workstations (3–10). Set cycle times individually or use uniform.',
            'Station Spacing': 'Meters between stations. Drives total walking distance.',
            'Walking Speed': 'Operator walking speed in m/s (default 1.2 — typical factory pace).',
            'Operators': 'Number of operators (1–4). Each gets a color-coded spaghetti trail.'
        },
        formula: 'Walk Distance/Cycle = Σ distances between assigned stations per operator route',
        feedsFrom: [
            { calcId: 'takt', label: 'Takt Time', key: 'takt' },
            { calcId: 'line-sim', label: 'Line Simulator (station times)' },
            { calcId: 'yamazumi', label: 'Yamazumi (station breakdown)' }
        ],
        feedsInto: [
            { calcId: 'oee', label: 'OEE (utilization data)' },
            { calcId: 'lineeff', label: 'Line Efficiency' }
        ],
        publishes: ['cellThroughput', 'cellWalkDist', 'cellUtilization']
    },
    'fmea-sim': {
        purpose: 'Monte Carlo simulation for FMEA risk assessment. Samples S/O/D ratings from uncertainty distributions (triangular) across thousands of scenarios. Reveals true risk distributions, compound system risk, and which failure modes drive the most overall exposure.',
        inputs: {
            'Failure Modes': 'List with Severity, Occurrence, Detection point estimates (1-10 each).',
            'Uncertainty Range': 'How much S/O/D can vary from the point estimate (±0.5 to ±3).',
            'Simulations': 'Number of Monte Carlo iterations (1,000 to 10,000).',
            'RPN Threshold': 'Flag scenarios where any mode exceeds this value.'
        },
        formula: 'RPN = S × O × D sampled from Triangular(rating ± uncertainty). System RPN = Σ per-mode RPNs.',
        feedsFrom: [
            { calcId: 'fmea', label: 'FMEA / RPN (failure mode data)' }
        ],
        feedsInto: [
            { calcId: 'riskmatrix', label: 'Risk Matrix' },
            { calcId: 'cpk', label: 'Cp / Cpk (capability check)' }
        ],
        publishes: ['fmeaSimPExceed', 'fmeaSimAvgRPN']
    },
    'smed-sim': {
        purpose: 'Interactive SMED (Single-Minute Exchange of Die) simulator. Reclassify changeover elements from internal (machine stopped) to external (machine running) and watch the downtime timeline shrink. Includes animated side-by-side before/after comparison.',
        inputs: {
            'Element Name': 'Description of the changeover task.',
            'Time': 'Minutes required for this element.',
            'Type': 'Internal (machine must stop) or External (can be done while running).'
        },
        formula: 'Downtime = Σ Internal element times. Reduction = (Before − After) ÷ Before × 100%',
        feedsFrom: [
            { calcId: 'smed', label: 'SMED Analysis (element data)' }
        ],
        feedsInto: [
            { calcId: 'heijunka-sim', label: 'Heijunka Sim (changeover time)' },
            { calcId: 'line-sim', label: 'Line Simulator (changeover param)' }
        ],
        publishes: ['smedSimBefore', 'smedSimAfter']
    },
    'heijunka-sim': {
        purpose: 'Side-by-side discrete-event simulation: batched production (all A then all B) vs leveled flow (ABABAB). Shows how leveling reduces WIP peaks and average lead time at the cost of more changeovers.',
        inputs: {
            'Product Mix': 'Products with demand quantities. Pull from the Heijunka calculator.',
            'Cycle Time': 'Seconds per unit of production.',
            'Changeover Time': 'Seconds lost when switching between products.',
            'Demand Interval': 'Seconds between customer pulls (mixed across products).'
        },
        formula: 'Leveled Sequence = smallest repeating demand pattern. WIP = units started − units completed.',
        feedsFrom: [
            { calcId: 'heijunka', label: 'Heijunka (product mix)' }
        ],
        feedsInto: [
            { calcId: 'smed', label: 'SMED (reduce changeover)' },
            { calcId: 'line-sim', label: 'Line Simulator' }
        ],
        publishes: ['hjSimBatchWip', 'hjSimLevelWip']
    },
    'safety-sim': {
        purpose: 'Animates the (s,Q) inventory policy day-by-day with stochastic demand and lead time. Shows actual stockouts, reorder triggers, and replenishment arrivals — making the safety stock formula tangible.',
        inputs: {
            'Avg Daily Demand': 'Units consumed per day on average.',
            'Demand Std Dev': 'Day-to-day demand variability (σ_d).',
            'Lead Time / Std Dev': 'Replenishment lead time and its variability (σ_LT).',
            'Service Level': 'Target probability of no stockout per cycle.',
            'Order Quantity': 'Units per replenishment order. Pull from EOQ.'
        },
        formula: 'SS = Z × √(LT·σ_d² + d²·σ_LT²)   ROP = d·LT + SS',
        feedsFrom: [
            { calcId: 'safety', label: 'Safety Stock Calculator', key: 'safetyStock' },
            { calcId: 'eoq', label: 'EOQ', key: 'eoq' }
        ],
        feedsInto: [],
        publishes: ['safetySimServiceLevel', 'safetySimAvgInventory', 'safetySimStockouts']
    },
    abc: {
        purpose: 'Segments inventory into A/B/C classes by cumulative value. Focus management attention on the vital few (A items) that drive 80% of value.',
        inputs: { 'SKU Data': 'Item names and annual usage values (paste CSV).', 'A Threshold': '% of cumulative value for A class (default 80%).', 'B Threshold': '% for B class (default 95%).' },
        formula: 'Sort by value desc → cumulative % → classify A/B/C',
        feedsFrom: [],
        feedsInto: [{ calcId: 'eoq', label: 'EOQ (for A items)', pullKey: 'abcAItems' }, { calcId: 'safety', label: 'Safety Stock', pullKey: 'abcAItems' }],
        publishes: ['abcAItems', 'abcBItems', 'abcCItems']
    },
    'demand-profile': {
        purpose: 'Classifies demand pattern using the Syntetos-Boylan framework. Determines whether demand is smooth, erratic, intermittent, or lumpy based on CoV² and ADI thresholds.',
        inputs: { 'Demand History': 'Comma-separated demand values per period. Zeros indicate no-demand periods.' },
        formula: 'CoV² = (σ/μ)² vs 0.49 threshold; ADI = avg inter-demand interval vs 1.32 threshold',
        feedsFrom: [],
        feedsInto: [{ calcId: 'safety', label: 'Safety Stock', pullKey: 'demandMean' }],
        publishes: ['demandPattern', 'demandMean', 'demandCoV']
    },
    'service-level': {
        purpose: 'Finds the optimal service level (fill rate) by balancing inventory holding costs against stockout costs. Shows the total cost curve across service levels.',
        inputs: { 'Mean Demand': 'Average demand per period.', 'Demand Std Dev': 'Demand variability.', 'Lead Time': 'Replenishment lead time.', 'Holding/Stockout Costs': 'Cost parameters for trade-off.' },
        formula: 'Optimal SL where marginal holding cost = marginal stockout cost reduction',
        feedsFrom: [{ calcId: 'safety', label: 'Safety Stock', key: 'safetyStock' }],
        feedsInto: [],
        publishes: ['optimalServiceLevel', 'optimalSafetyStock']
    },
    mrp: {
        purpose: 'Performs gross-to-net requirements explosion. Starting from gross requirements and on-hand inventory, calculates net requirements and offsets planned order releases by lead time.',
        inputs: { 'Gross Requirements': 'Demand per period.', 'Scheduled Receipts': 'Already-ordered quantities.', 'On-Hand': 'Starting inventory.', 'Lead Time': 'Order lead time.' },
        formula: 'Net = Gross - On Hand - Receipts + Safety Stock; Planned Release = Net offset by LT',
        feedsFrom: [{ calcId: 'eoq', label: 'EOQ', key: 'eoq' }],
        feedsInto: [],
        publishes: ['mrpPlannedOrders']
    }
};

/**
 * Lookup table: (calcId, publishKey) → target input element ID.
 * Maps the 15 known pull-button connections.
 */
function findPullTarget(calcId, publishKey) {
    const pullMap = {
        'rto':            { takt: 'rto-takt' },
        'yamazumi':       { takt: 'yama-takt' },
        'line-sim':       { takt: 'ls-takt', changeoverInternal: 'ls-changeover-time' },
        'kanban':         { eoq: 'kanban-container' },
        'epei':           { changeoverInternal: 'epei-changeover' },
        'oee':            { bottleneckCT: 'oee-ideal' },
        'littles':        { bottleneckThroughput: 'littles-thr' },
        'queue':          { bottleneckThroughput: 'queue-mu' },
        'queue-priority': { bottleneckThroughput: 'qp-mu' },
        'pitch':          { takt: 'pitch-takt' },
        'turns':          { kanbanInventory: 'turns-inv' },
        'lineeff':        { changeoverInternal: 'lineeff-changeover' },
        'heijunka':       { pitch: 'heijunka-pitch' },
        'safety-sim':     { eoq: 'ss-order-qty' }
    };
    return pullMap[calcId]?.[publishKey] || null;
}

/**
 * Build context-aware prompts by inspecting SvendOps.values.
 * Returns [{text, action}] — zero API cost.
 */
function buildSmartPrompts(id, guide) {
    const prompts = [];
    const linkIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>';

    // Upstream data available? Offer to pull it in.
    if (guide.feedsFrom) {
        guide.feedsFrom.forEach(f => {
            const val = SvendOps.values[f.key];
            if (val) {
                const target = findPullTarget(id, f.key);
                if (target) {
                    prompts.push({
                        text: `${f.label} (${val.value} ${val.unit}) is available — pull it in`,
                        action: `SvendOps.pull('${f.key}','${target}')`
                    });
                } else {
                    prompts.push({
                        text: `${f.label} has been calculated (${val.value} ${val.unit})`,
                        action: `navigateToCalc('${f.calcId}')`
                    });
                }
            }
        });
    }

    // Suggest upstream if nothing available yet
    if (prompts.length === 0 && guide.feedsFrom) {
        guide.feedsFrom.forEach(f => {
            if (!SvendOps.values[f.key]) {
                prompts.push({
                    text: `Consider calculating ${f.label} first for a connected workflow`,
                    action: `navigateToCalc('${f.calcId}')`
                });
            }
        });
    }

    return prompts;
}

/**
 * Render the semi-smart guide widget for a calculator.
 * Called from showCalc(). Single container, dynamically populated.
 */
function renderGuide(id) {
    const container = document.getElementById('calc-guide');
    const guide = calcGuide[id];

    if (!guide) {
        container.classList.remove('visible');
        return;
    }

    const chevron = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>';

    // 1. Purpose
    let html = `<div class="calc-guide-section">
        <div class="calc-guide-section-title">What This Does</div>
        <p>${guide.purpose}</p>
    </div>`;

    // 2. Inputs
    if (guide.inputs) {
        const lines = Object.entries(guide.inputs)
            .map(([name, desc]) => `<strong>${name}</strong> — ${desc}`)
            .join('<br>');
        html += `<div class="calc-guide-section">
            <div class="calc-guide-section-title">Inputs</div>
            <p>${lines}</p>
        </div>`;
    }

    // 3. Formula
    if (guide.formula) {
        html += `<div class="calc-guide-section">
            <div class="calc-guide-section-title">Formula</div>
            <p style="font-family:'SF Mono','Fira Code',monospace; color:var(--accent);">${guide.formula}</p>
        </div>`;
    }

    // 4. Workflow context
    const hasFrom = guide.feedsFrom?.length > 0;
    const hasInto = guide.feedsInto?.length > 0;
    if (hasFrom || hasInto) {
        let flow = '<div class="calc-guide-flow">';
        if (hasFrom) {
            flow += `<div class="calc-guide-flow-col">
                <div class="calc-guide-section-title">Pulls From</div>
                <ul>${guide.feedsFrom.map(f =>
                    `<li onclick="navigateToCalc('${f.calcId}')">${f.label}</li>`
                ).join('')}</ul>
            </div>`;
        }
        if (hasInto) {
            flow += `<div class="calc-guide-flow-col">
                <div class="calc-guide-section-title">Feeds Into</div>
                <ul>${guide.feedsInto.map(f =>
                    `<li onclick="navigateToCalc('${f.calcId}')">${f.label}</li>`
                ).join('')}</ul>
            </div>`;
        }
        flow += '</div>';
        html += `<div class="calc-guide-section">${flow}</div>`;
    }

    // 5. Smart prompts
    const prompts = buildSmartPrompts(id, guide);
    if (prompts.length > 0) {
        html += `<div class="calc-guide-prompts">
            ${prompts.map(p => `<div class="calc-guide-prompt" onclick="${p.action}">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>
                <span>${p.text}</span>
            </div>`).join('')}
        </div>`;
    }

    // Preserve expand state
    const wasExpanded = container.querySelector('.calc-guide-card')?.classList.contains('expanded');

    container.innerHTML = `<div class="calc-guide-card${wasExpanded ? ' expanded' : ''}">
        <div class="calc-guide-header" onclick="this.parentElement.classList.toggle('expanded')">
            ${chevron}
            Guide
        </div>
        <div class="calc-guide-body">${html}</div>
    </div>`;
    container.classList.add('visible');
}

/**
 * Render "Next Steps" cross-link cards after a calculation completes.
 * @param {string} containerId  - ID of the container div (e.g. 'takt-next-steps')
 * @param {Array<{title:string, desc:string, calcId:string, pullKey?:string, pullTarget?:string}>} steps
 */
function renderNextSteps(containerId, steps) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const chevron = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18l6-6-6-6"/></svg>';
    container.innerHTML = steps.map(s => {
        const pull = s.pullKey && s.pullTarget
            ? `SvendOps.pull('${s.pullKey}','${s.pullTarget}');`
            : '';
        return `<div class="calc-next-step" onclick="${pull}navigateToCalc('${s.calcId}')">
            ${chevron}
            <div>
                <div class="calc-next-step-title">${s.title}</div>
                <div class="calc-next-step-desc">${s.desc}</div>
            </div>
        </div>`;
    }).join('');
    container.style.display = 'flex';
}

/**
 * Navigate to a calculator by ID — works without event context.
 * Clicks the correct nav button to trigger showCalc properly.
 */
function navigateToCalc(calcId) {
    const btn = document.querySelector(`.ops-nav-item[onclick*="'${calcId}'"]`);
    if (btn) { btn.click(); btn.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }
}

// ============================================================================
// Custom Stepper Widget
// ============================================================================

/**
 * Convert a number input to a custom stepper widget
 * Preserves all original attributes and behaviors
 */
function createStepper(input, options = {}) {
    // Skip if already converted
    if (input.dataset.stepperized) return;
    input.dataset.stepperized = 'true';

    const step = parseFloat(input.step) || 1;
    const min = input.hasAttribute('min') ? parseFloat(input.min) : null;
    const max = input.hasAttribute('max') ? parseFloat(input.max) : null;
    const isSmall = options.small || false;

    // Create wrapper
    const wrapper = document.createElement('div');
    wrapper.className = 'stepper' + (isSmall ? ' stepper-sm' : '');

    // Minus button
    const minusBtn = document.createElement('button');
    minusBtn.type = 'button';
    minusBtn.className = 'stepper-btn';
    minusBtn.textContent = '−';
    minusBtn.tabIndex = -1;

    // Plus button
    const plusBtn = document.createElement('button');
    plusBtn.type = 'button';
    plusBtn.className = 'stepper-btn';
    plusBtn.textContent = '+';
    plusBtn.tabIndex = -1;

    // Move input styling
    input.className = 'stepper-value';

    // Build stepper
    input.parentNode.insertBefore(wrapper, input);
    wrapper.appendChild(minusBtn);
    wrapper.appendChild(input);
    wrapper.appendChild(plusBtn);

    // Button handlers
    function adjust(delta) {
        let val = parseFloat(input.value) || 0;
        val += delta;
        if (min !== null) val = Math.max(min, val);
        if (max !== null) val = Math.min(max, val);
        // Round to avoid float errors
        val = Math.round(val * 1e10) / 1e10;
        input.value = val;
        input.dispatchEvent(new Event('input', { bubbles: true }));
    }

    minusBtn.addEventListener('click', (e) => {
        e.preventDefault();
        adjust(-step);
    });

    plusBtn.addEventListener('click', (e) => {
        e.preventDefault();
        adjust(step);
    });

    // Hold-to-repeat
    let holdTimer = null;
    let holdInterval = null;

    function startHold(delta) {
        holdTimer = setTimeout(() => {
            holdInterval = setInterval(() => adjust(delta), 75);
        }, 400);
    }

    function stopHold() {
        clearTimeout(holdTimer);
        clearInterval(holdInterval);
    }

    [minusBtn, plusBtn].forEach((btn, i) => {
        const delta = i === 0 ? -step : step;
        btn.addEventListener('mousedown', () => startHold(delta));
        btn.addEventListener('mouseup', stopHold);
        btn.addEventListener('mouseleave', stopHold);
        btn.addEventListener('touchstart', (e) => { e.preventDefault(); startHold(delta); });
        btn.addEventListener('touchend', stopHold);
    });

    return wrapper;
}

/**
 * Convert all number inputs in calc-input containers to steppers
 * Run this on page load
 */
function initializeSteppers() {
    // Main calc inputs (larger steppers)
    document.querySelectorAll('.calc-input input[type="number"]').forEach(input => {
        createStepper(input);
    });
}

// ============================================================================
// Monte Carlo Simulation Engine
// ============================================================================

const MonteCarlo = {
    // Run simulation with variability
    simulate(calcFn, inputs, iterations = 2000) {
        const results = [];
        for (let i = 0; i < iterations; i++) {
            // Add random variability to each input
            const variedInputs = inputs.map(inp => {
                const variability = inp.cv || 0.1; // Default ±10%
                const noise = 1 + (Math.random() - 0.5) * 2 * variability;
                return inp.value * noise;
            });
            results.push(calcFn(...variedInputs));
        }
        return this.analyze(results);
    },

    // Analyze simulation results
    analyze(results) {
        results.sort((a, b) => a - b);
        const n = results.length;
        const mean = results.reduce((a, b) => a + b, 0) / n;
        const variance = results.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
        const std = Math.sqrt(variance);

        return {
            mean,
            std,
            min: results[0],
            max: results[n - 1],
            p5: results[Math.floor(n * 0.05)],
            p25: results[Math.floor(n * 0.25)],
            p50: results[Math.floor(n * 0.50)],
            p75: results[Math.floor(n * 0.75)],
            p95: results[Math.floor(n * 0.95)],
            histogram: this.buildHistogram(results, 20),
            raw: results
        };
    },

    // Build histogram buckets
    buildHistogram(results, buckets) {
        const min = results[0];
        const max = results[results.length - 1];
        const range = max - min || 1;
        const bucketSize = range / buckets;

        const hist = new Array(buckets).fill(0);
        results.forEach(r => {
            const idx = Math.min(buckets - 1, Math.floor((r - min) / bucketSize));
            hist[idx]++;
        });

        return {
            counts: hist,
            labels: hist.map((_, i) => (min + (i + 0.5) * bucketSize).toFixed(1))
        };
    },

    // Render histogram chart
    renderHistogram(containerId, simResult, title, unit) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const hist = simResult.histogram;

        ForgeViz.render(container, {
            title: title, chart_type: 'bar',
            traces: [{ x: hist.labels, y: hist.counts, name: '', trace_type: 'bar',
                color: 'rgba(74, 159, 110, 0.7)' }],
            reference_lines: [
                { value: simResult.p5, axis: 'x', color: '#e74c3c', dash: 'dashed', label: '5th %ile' },
                { value: simResult.p95, axis: 'x', color: '#e74c3c', dash: 'dashed', label: '95th %ile' }
            ],
            markers: [{ x: simResult.mean, y: Math.max(...hist.counts) * 0.9,
                label: `μ = ${simResult.mean.toFixed(1)}`, color: '#4a9f6e' }],
            zones: [],
            x_axis: { label: unit }, y_axis: { label: 'Frequency' }
        });
    }
};

// ============================================================================
// Derivation Helper — Show Your Work
// ============================================================================

function toggleDerivation(id) {
    const el = document.getElementById(id);
    el.classList.toggle('expanded');
}

let currentCalcId = 'takt';

function filterCalcs(query) {
    const q = query.toLowerCase().trim();
    document.querySelectorAll('.ops-nav-group').forEach(group => {
        const items = group.querySelectorAll('.ops-nav-item');
        let anyVisible = false;
        items.forEach(item => {
            const label = item.querySelector('span')?.textContent.toLowerCase() || '';
            const id = (item.getAttribute('onclick') || '').match(/showCalc\('([^']+)'\)/)?.[1] || '';
            const meta = calcMeta[id];
            const desc = meta?.desc?.toLowerCase() || '';
            const match = !q || label.includes(q) || desc.includes(q) || id.includes(q);
            item.classList.toggle('hidden', !match);
            if (match) anyVisible = true;
        });
        group.classList.toggle('hidden', !anyVisible);
    });
}

function showCalc(id) {
    currentCalcId = id;
    // Update nav
    document.querySelectorAll('.ops-nav-item').forEach(el => el.classList.remove('active'));
    event.currentTarget.classList.add('active');

    // Update header
    document.getElementById('calc-title').textContent = calcMeta[id].title;
    document.getElementById('calc-desc').textContent = calcMeta[id].desc;

    // Show layout
    document.querySelectorAll('.calc-layout').forEach(el => el.classList.remove('active'));
    document.getElementById(`layout-${id}`).classList.add('active');

    // Render guide widget
    renderGuide(id);
}
