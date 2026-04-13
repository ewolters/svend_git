// =============================================================================
// Guided Scenarios — optional teaching mode
// =============================================================================

let activeScenario = null;
let activeStepIndex = 0;
let scenarioProgress = {};
let scenarioChallengeResults = null;

function loadScenarioProgress() {
    try { scenarioProgress = JSON.parse(localStorage.getItem('svend_scenario_progress') || '{}'); }
    catch { scenarioProgress = {}; }
}
function saveScenarioProgress() {
    localStorage.setItem('svend_scenario_progress', JSON.stringify(scenarioProgress));
}
loadScenarioProgress();

const GUIDED_SCENARIOS = [
    // =========================================================================
    // TIER 1: FOUNDATIONS
    // =========================================================================
    {
        id: 'single-machine',
        title: 'The Single Machine',
        difficulty: 1,
        estimatedMinutes: 10,
        category: 'foundations',
        features: ['cycle time', 'throughput', "Little's Law"],
        overview: "Understand cycle time, throughput, and what variability does to a supposedly simple system.",
        layout: {
            stations: [
                { id: 'stn-1', type: 'single', name: 'Press', x: 350, y: 220, cycle_time: 30, cycle_time_cv: 0, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
            ],
            sources: [
                { id: 'src-1', name: 'Raw Material', x: 120, y: 220, arrival_distribution: 'fixed', arrival_rate: 30, arrival_cv: 0, batch_size: 1, product_types: [{ name: 'A', ratio: 1.0, color: '#4a9f6e' }], schedule_mode: 'fixed_mix' },
            ],
            sinks: [
                { id: 'sink-1', name: 'Finished Goods', x: 580, y: 220, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'conn-1', from_id: 'src-1', to_id: 'stn-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-2', from_id: 'stn-1', to_id: 'sink-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 3600, speed: 50 },
        steps: [
            {
                title: 'Observe the Baseline',
                instruction: 'Hit Play and let the simulation run to completion. Watch the throughput metric in the left panel.',
                teaching: 'This machine has a 30-second cycle time with zero variability. In theory: 3600/30 = 120 parts/hour. The warmup period (first 300s) is excluded from metrics so the numbers are clean. See if reality matches theory.',
                highlightElements: ['m-throughput'],
                highlightSections: ['Controls', 'Metrics'],
            },
            {
                title: 'Reduce the Cycle Time',
                instruction: 'Click the Press machine on the canvas, then change its cycle time to 20s in the Properties panel. Reset and run again.',
                teaching: "You just increased capacity by 50% on paper. In a real factory, cycle time reduction is the hardest win — it usually requires capital investment, engineering changes, or process redesign. That's why Goldratt's first rule is to exploit the constraint before elevating it. Faster machines cost money. Removing waste from the existing process is free.",
                highlightSections: ['Properties', 'Controls'],
            },
            {
                title: 'Add Variability',
                instruction: 'Set the cycle time CV (coefficient of variation) to 0.30 (30%). Reset and run again. Compare your throughput to the previous run.',
                teaching: "With CV=30%, the average cycle time is still 20s, but individual jobs vary. Your throughput drops even though the mean hasn't changed. This is Kingman's formula at work: as variability increases, queue time grows exponentially, not linearly. This is why reducing variation — through standardized work, TPM, mistake-proofing — is often more valuable than reducing the average.",
                highlightSections: ['Properties', 'Controls', 'Metrics'],
            },
            {
                title: "Verify Little's Law",
                instruction: "Look at the WIP, throughput, and lead time metrics. Check: does WIP ≈ throughput × lead time? (Remember throughput is in parts/hour, lead time is in seconds — convert units.)",
                teaching: "Little's Law (L = λW) is the only universal law in queuing theory. It always holds in steady state, regardless of distribution, service discipline, or number of servers. WIP = throughput rate × average time in system. If you want to reduce lead time, you must either reduce WIP or increase throughput. There are no other levers. This is the foundation everything else builds on.",
                highlightElements: ['m-wip', 'm-throughput', 'm-leadtime'],
            },
        ],
        challenges: [
            { description: 'Achieve throughput above 100 parts/hour', evaluate: (r) => r.throughput >= 100 },
            { description: 'Run with CV > 0 (variability present)', evaluate: (r) => r.avg_lead_time > 0 && r.throughput < 180 },
        ],
    },
    {
        id: 'bottleneck',
        title: 'The Bottleneck',
        difficulty: 1,
        estimatedMinutes: 12,
        category: 'foundations',
        features: ['bottleneck', 'TOC', 'WIP buildup'],
        overview: "Discover that a system is only as fast as its slowest machine. The foundation of the Theory of Constraints.",
        layout: {
            stations: [
                { id: 'stn-1', type: 'single', name: 'Cut', x: 280, y: 220, cycle_time: 20, cycle_time_cv: 0.1, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
                { id: 'stn-2', type: 'single', name: 'Weld', x: 480, y: 220, cycle_time: 35, cycle_time_cv: 0.1, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
            ],
            sources: [
                { id: 'src-1', name: 'Raw Material', x: 80, y: 220, arrival_distribution: 'exponential', arrival_rate: 25, arrival_cv: 0, batch_size: 1, product_types: [{ name: 'A', ratio: 1.0, color: '#4a9f6e' }], schedule_mode: 'fixed_mix' },
            ],
            sinks: [
                { id: 'sink-1', name: 'Finished Goods', x: 680, y: 220, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'conn-1', from_id: 'src-1', to_id: 'stn-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-2', from_id: 'stn-1', to_id: 'stn-2', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-3', from_id: 'stn-2', to_id: 'sink-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 3600, speed: 50 },
        steps: [
            {
                title: 'Find the Bottleneck',
                instruction: 'Run the simulation. Watch where WIP accumulates — the queue will pile up before the slowest machine. Check which machine the Bottleneck metric names.',
                teaching: "Cut runs at 20s/part (180/hr capacity). Weld runs at 35s/part (103/hr capacity). Material arrives every 25s (144/hr). The system can never produce faster than the Weld station's 103/hr, no matter how fast Cut runs. WIP stacks up between them because Cut produces faster than Weld can consume.",
                highlightElements: ['m-bottleneck', 'm-wip'],
            },
            {
                title: 'Speed Up the Wrong Machine',
                instruction: 'Click Cut and reduce its cycle time to 10s. Reset and run. Does system throughput increase?',
                teaching: "No. Throughput stays ~103/hr because Weld is still the constraint. You just made Cut faster at producing WIP. Improving a non-bottleneck is waste — it only increases the pile of inventory between stations. This is Goldratt's core insight: any improvement not at the constraint is an illusion.",
                highlightSections: ['Properties', 'Metrics'],
            },
            {
                title: 'Fix the Real Constraint',
                instruction: "Now click Weld and reduce its cycle time to 20s. Reset and run. What happens to throughput and WIP?",
                teaching: "Throughput jumps because you addressed the actual constraint. WIP drops because the bottleneck can now keep pace with upstream flow. In practice, you'd first exploit the constraint (ensure it never starves, never idles, never processes rework), then subordinate everything else to it, then elevate (add capacity) only if needed. That's the TOC 5 Focusing Steps.",
                highlightElements: ['m-throughput', 'm-wip'],
            },
            {
                title: 'The New Bottleneck',
                instruction: "With both machines at 20s, the constraint has moved. Where is it now? Try reducing arrival rate to 30s and observe.",
                teaching: "When you break one constraint, another emerges. The constraint might shift to the arrival rate (demand), to a different machine, or to a policy (batch rules, shift schedule). The system always has exactly one constraint governing throughput. The art of operations is knowing which one it is right now.",
            },
        ],
        challenges: [
            { description: 'Achieve system throughput above 120 parts/hour', evaluate: (r) => r.throughput >= 120 },
            { description: 'Keep average WIP below 8 parts', evaluate: (r) => r.avg_wip < 8 },
        ],
    },
    {
        id: 'buffers-flow',
        title: 'Buffers and Flow',
        difficulty: 2,
        estimatedMinutes: 15,
        category: 'foundations',
        features: ['buffers', 'blocking', 'starving', 'WIP'],
        overview: "Learn why buffers exist, what happens without them, and why bigger is not always better.",
        layout: {
            stations: [
                { id: 'stn-1', type: 'single', name: 'Machine A', x: 260, y: 220, cycle_time: 25, cycle_time_cv: 0.25, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
                { id: 'stn-2', type: 'single', name: 'Machine B', x: 520, y: 220, cycle_time: 25, cycle_time_cv: 0.25, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
            ],
            sources: [
                { id: 'src-1', name: 'Material In', x: 60, y: 220, arrival_distribution: 'exponential', arrival_rate: 25, arrival_cv: 0, batch_size: 1, product_types: [{ name: 'A', ratio: 1.0, color: '#4a9f6e' }], schedule_mode: 'fixed_mix' },
            ],
            sinks: [
                { id: 'sink-1', name: 'Ship', x: 720, y: 220, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'conn-1', from_id: 'src-1', to_id: 'stn-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-2', from_id: 'stn-1', to_id: 'stn-2', buffer_capacity: 5, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-3', from_id: 'stn-2', to_id: 'sink-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 7200, speed: 50 },
        steps: [
            {
                title: 'Run With a Small Buffer',
                instruction: 'The connection between Machine A and Machine B has a buffer capacity of 5. Run the simulation and observe blocking/starving times in the utilization charts.',
                teaching: "Both machines have identical cycle times (25s) but CV=25% — they're variable. Sometimes A finishes faster and has to wait (blocked) because B's queue is full. Sometimes B finishes faster and has to wait (starved) because nothing is in the buffer. The buffer decouples their variability.",
                highlightSections: ['Controls', 'Metrics'],
            },
            {
                title: 'Remove the Buffer',
                instruction: 'Click the connection between A and B. Set buffer capacity to 1 (minimum). Reset and run. Watch blocking and starving skyrocket.',
                teaching: "With a buffer of 1, every time the two machines get out of sync, one must wait. Blocking at A means it finishes a part but can't release it — lost capacity. Starving at B means it sits idle — also lost capacity. With no buffer to absorb timing mismatches, you lose throughput to synchronization waste.",
                highlightSections: ['Properties'],
            },
            {
                title: 'Oversize the Buffer',
                instruction: 'Set the buffer to 50. Reset and run. Throughput recovers — but look at average WIP and lead time.',
                teaching: "The big buffer eliminates blocking and starving — throughput is good. But now you're carrying 15-25 parts of WIP between stations. Each part sits in that buffer, aging, tying up cash, hiding quality problems. If Machine A starts producing defects, you won't know until 50 parts later. Large buffers are a crutch — they mask problems instead of solving them.",
            },
            {
                title: 'Find the Right Size',
                instruction: "Try buffer sizes of 3, 5, 8, 10. For each, note throughput, WIP, and lead time. Find the smallest buffer that maintains good throughput.",
                teaching: "The optimal buffer is the minimum needed to absorb normal variability without significant blocking or starving. This is the fundamental tension in lean manufacturing: buffers provide stability but create waste. Toyota's answer is to reduce variability (through TPM, standardized work, heijunka) so you can shrink buffers without losing throughput. Don't start by cutting buffers — start by cutting the variability that makes them necessary.",
                highlightElements: ['m-throughput', 'm-wip', 'm-leadtime'],
            },
        ],
        challenges: [
            { description: 'Achieve throughput above 120/hr with buffer ≤ 8', evaluate: (r) => r.throughput >= 120 && r.avg_wip < 15 },
            { description: 'Keep average lead time below 120s', evaluate: (r) => r.avg_lead_time < 120 },
        ],
    },

    // =========================================================================
    // TIER 2: VARIABILITY & RELIABILITY
    // =========================================================================
    {
        id: 'breakdowns',
        title: 'Breakdowns and Reliability',
        difficulty: 2,
        estimatedMinutes: 15,
        category: 'variability',
        features: ['MTBF', 'MTTR', 'Weibull', 'PM'],
        overview: "Machines break. Learn how failure patterns affect throughput and why predictive maintenance matters.",
        layout: {
            stations: [
                { id: 'stn-1', type: 'single', name: 'Reliable', x: 260, y: 150, cycle_time: 25, cycle_time_cv: 0.1, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
                { id: 'stn-2', type: 'single', name: 'Unreliable', x: 260, y: 300, cycle_time: 25, cycle_time_cv: 0.1, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: 1800, mttr: 300, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1 },
                { id: 'stn-3', type: 'single', name: 'Assembly', x: 520, y: 220, cycle_time: 20, cycle_time_cv: 0.1, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
            ],
            sources: [
                { id: 'src-1', name: 'Material', x: 60, y: 220, arrival_distribution: 'exponential', arrival_rate: 20, arrival_cv: 0, batch_size: 1, product_types: [{ name: 'A', ratio: 1.0, color: '#4a9f6e' }], schedule_mode: 'fixed_mix' },
            ],
            sinks: [
                { id: 'sink-1', name: 'Finished Goods', x: 720, y: 220, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'conn-1', from_id: 'src-1', to_id: 'stn-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-2', from_id: 'src-1', to_id: 'stn-2', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-3', from_id: 'stn-1', to_id: 'stn-3', buffer_capacity: 10, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-4', from_id: 'stn-2', to_id: 'stn-3', buffer_capacity: 10, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-5', from_id: 'stn-3', to_id: 'sink-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 7200, speed: 50 },
        steps: [
            {
                title: 'See the Impact',
                instruction: 'Run the simulation. Both paths feed Assembly, but "Unreliable" has MTBF=1800s, MTTR=300s. Watch the queue charts — one path will be erratic.',
                teaching: "MTBF 1800s = breaks down on average every 30 minutes. MTTR 300s = takes 5 minutes to fix. Theoretical availability = 1800/(1800+300) = 85.7%. But that 14.3% downtime doesn't distribute evenly — it comes in lumps. When the machine goes down, everything downstream of it starves. The variance of the loss matters more than the average.",
                highlightElements: ['m-bottleneck'],
            },
            {
                title: 'Increase Repair Speed',
                instruction: 'Click Unreliable, change MTTR to 120s (faster repair). Reset and run. How much does throughput improve?',
                teaching: "Reducing MTTR is almost always cheaper than increasing MTBF. MTTR reduction means: spare parts staged at the machine, trained operators who can diagnose fast, standardized repair procedures, quick-change modules. This is the heart of TPM (Total Productive Maintenance). You can't prevent all failures, but you can control how fast you recover.",
            },
            {
                title: 'Change the Failure Pattern',
                instruction: "Click Unreliable and look for Weibull Beta. Beta=1 is random failures. Set it to 2 (wear-out pattern). Reset and run.",
                teaching: "With beta=1 (exponential), failures are memoryless — the machine is equally likely to fail whether it ran 5 minutes or 5 hours. You can't predict them. With beta=2 (Weibull wear-out), failure probability increases with run time — you can schedule PM before the failure happens. This is why condition monitoring and predictive maintenance work for mechanical wear but not for random electrical failures.",
            },
            {
                title: 'Add Preventive Maintenance',
                instruction: 'Set a PM interval on Unreliable (try 1500s with 60s PM duration). This stops the machine periodically for planned maintenance. Compare to the no-PM run.',
                teaching: "PM trades small planned stops for large unplanned ones. If PM interval is shorter than typical time-to-failure, you catch wear before it causes a breakdown. The catch: PM itself costs production time. Over-maintaining is waste too. The optimal PM interval depends on the Weibull beta — higher beta (more predictable wear) means PM is more effective. For beta=1 (random), PM doesn't help at all.",
                highlightSections: ['Properties'],
            },
        ],
        challenges: [
            { description: 'Achieve throughput above 140/hr despite breakdowns', evaluate: (r) => r.throughput >= 140 },
            { description: 'Keep machine downtime below 15% on the unreliable machine', evaluate: (r) => { const utils = r.station_utilizations || {}; for (const u of Object.values(utils)) { if (u.down > 0.15) return false; } return true; } },
        ],
    },
    {
        id: 'changeovers',
        title: 'Changeovers and Product Mix',
        difficulty: 2,
        estimatedMinutes: 15,
        category: 'variability',
        features: ['SMED', 'changeover', 'batch size', 'product mix'],
        overview: "Every product switch costs time. Learn the SMED philosophy and why smaller batches are almost always better.",
        layout: {
            stations: [
                { id: 'stn-1', type: 'single', name: 'Press', x: 350, y: 220, cycle_time: 20, cycle_time_cv: 0.1, changeover_time: 180, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
            ],
            sources: [
                { id: 'src-1', name: 'Orders', x: 100, y: 220, arrival_distribution: 'exponential', arrival_rate: 25, arrival_cv: 0, batch_size: 1, product_types: [{ name: 'Widget', ratio: 0.6, color: '#4a9f6e' }, { name: 'Gadget', ratio: 0.4, color: '#3b82f6' }], schedule_mode: 'fixed_mix' },
            ],
            sinks: [
                { id: 'sink-1', name: 'Finished Goods', x: 600, y: 220, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'conn-1', from_id: 'src-1', to_id: 'stn-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-2', from_id: 'stn-1', to_id: 'sink-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 7200, speed: 50 },
        steps: [
            {
                title: 'See the Cost of Changeovers',
                instruction: 'Run the simulation. The Press makes 2 products (Widget 60%, Gadget 40%). Every product switch costs 180 seconds (3 minutes). Watch the setup time accumulate in the utilization chart.',
                teaching: "With random arrival mix, changeovers happen frequently — potentially every few parts. At 180s per changeover, you might lose 20-30% of production time to setup. This is the hidden factory: the machine runs fine when it's running, but it's not running enough. Most factories track uptime but not setup loss separately.",
                highlightSections: ['Controls', 'Metrics'],
            },
            {
                title: 'Apply SMED',
                instruction: 'Click the Press, reduce changeover time to 30s (aggressive but achievable SMED). Reset and run.',
                teaching: "SMED (Single-Minute Exchange of Die) was developed by Shigeo Shingo at Toyota. The method: separate internal setup (machine must be stopped) from external setup (can be done while running). Convert internal to external. Then streamline what's left. Going from 180s to 30s is a real-world result — Fort Dearborn cut Heidelberg press makereadies from hours to under 15 minutes using exactly this approach.",
            },
            {
                title: 'Try Batch Sequencing',
                instruction: "Change the source's schedule mode to batch_sequence. Set batch size to 10 for each product. Reset and run. Fewer changeovers, but watch WIP and lead time.",
                teaching: "Batching reduces changeover frequency — produce 10 Widgets, then switch to 10 Gadgets. Fewer changeovers = more capacity. But: larger batches mean longer lead times (Gadget orders wait while Widget batch completes), more WIP, and longer time to detect quality problems. The economic batch quantity balances changeover cost against holding cost. Lean's answer is to attack changeover time directly so the optimal batch size approaches 1.",
                highlightElements: ['m-wip', 'm-leadtime'],
            },
            {
                title: 'Find Your Balance',
                instruction: 'Experiment with different combinations: changeover time (30-180s) and batch size (1, 5, 10, 20). For each, note throughput, WIP, and lead time.',
                teaching: "There's no free lunch. Long changeovers force large batches. Large batches cause high WIP. High WIP causes long lead times. Long lead times cause due date misses. The only way to break the cycle is to attack changeover time. Once changeovers are fast enough, you can run batch size = 1 (one-piece flow) and get low WIP, short lead times, and flexible response to demand changes. That's the lean ideal.",
            },
        ],
        challenges: [
            { description: 'Achieve throughput above 100/hr with 2 product types', evaluate: (r) => r.throughput >= 100 },
            { description: 'Keep average lead time below 90 seconds', evaluate: (r) => r.avg_lead_time < 90 },
        ],
    },
    {
        id: 'workforce',
        title: 'Workforce Constraints',
        difficulty: 3,
        estimatedMinutes: 15,
        category: 'variability',
        features: ['operators', 'skills', 'cross-training', 'calloff'],
        overview: "Machines don't run themselves. Learn why cross-training is insurance and what happens when people call off.",
        layout: {
            stations: [
                { id: 'stn-1', type: 'single', name: 'CNC Lathe', x: 280, y: 150, cycle_time: 30, cycle_time_cv: 0.15, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 1, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
                { id: 'stn-2', type: 'single', name: 'Mill', x: 280, y: 300, cycle_time: 30, cycle_time_cv: 0.15, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 1, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
                { id: 'stn-3', type: 'single', name: 'Grind', x: 520, y: 220, cycle_time: 25, cycle_time_cv: 0.15, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 1, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
            ],
            sources: [
                { id: 'src-1', name: 'Material', x: 60, y: 220, arrival_distribution: 'exponential', arrival_rate: 20, arrival_cv: 0, batch_size: 1, product_types: [{ name: 'Part', ratio: 1.0, color: '#4a9f6e' }], schedule_mode: 'fixed_mix' },
            ],
            sinks: [
                { id: 'sink-1', name: 'Ship', x: 720, y: 220, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'conn-1', from_id: 'src-1', to_id: 'stn-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-2', from_id: 'src-1', to_id: 'stn-2', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-3', from_id: 'stn-1', to_id: 'stn-3', buffer_capacity: 8, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-4', from_id: 'stn-2', to_id: 'stn-3', buffer_capacity: 8, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-5', from_id: 'stn-3', to_id: 'sink-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [
                { id: 'op-1', name: 'Alex', skills: { 'stn-1': 1.0, 'stn-2': 0, 'stn-3': 0 }, calloffRate: 0, status: 'available' },
                { id: 'op-2', name: 'Blake', skills: { 'stn-1': 0, 'stn-2': 1.0, 'stn-3': 0 }, calloffRate: 0, status: 'available' },
                { id: 'op-3', name: 'Casey', skills: { 'stn-1': 0, 'stn-2': 0, 'stn-3': 1.0 }, calloffRate: 0, status: 'available' },
            ],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 14400, speed: 100 },
        steps: [
            {
                title: 'Observe the Baseline',
                instruction: 'Run with 3 operators, each trained on exactly one machine. Throughput should be decent. Now open the Workforce section and note the skill assignments.',
                teaching: "Three machines, three operators, each specialized. The system works perfectly — as long as everyone shows up. This is the brittle equilibrium most factories operate in. It looks efficient because everyone is busy, but it has zero resilience to disruption.",
                highlightSections: ['Workforce', 'Controls'],
            },
            {
                title: 'Simulate an Absence',
                instruction: "Set Alex's calloff rate to 100% (will always be absent). Reset and run. CNC Lathe has no operator — it can't run.",
                teaching: "One person calls off and an entire production line stalls. This happens in real factories every single day. The CNC Lathe sits idle because nobody else knows how to run it. The other two operators are still working, but the system throughput collapses because the bottleneck is now the unmanned machine. This is the cost of specialization.",
                highlightSections: ['Workforce'],
            },
            {
                title: 'Cross-Train',
                instruction: "Give Blake a skill on CNC Lathe (use the Cross-Train button or edit skills directly). Set Alex's calloff to 15% (realistic). Set Blake's calloff to 15% too. Reset and run.",
                teaching: "Now when Alex is absent, Blake can cover CNC Lathe — though Blake has to choose between CNC Lathe and Mill. The system degrades gracefully instead of collapsing. Cross-training costs real production time (training hours) but provides insurance against the most common disruption in manufacturing: people not showing up.",
            },
            {
                title: 'The Insurance Policy',
                instruction: "Add a 4th operator (+ Operator button). Cross-train them on all 3 machines. Set all calloff rates to 15%. Run several times and compare throughput consistency.",
                teaching: "The 4th operator is insurance. On a normal day, they float to wherever the queue is longest. On an absence day, they cover the gap. This costs labor (you're paying someone who might not always be needed), but the alternative is unpredictable throughput, missed deliveries, and overtime. Cross-trained floaters are the cheapest insurance in manufacturing. Every shift needs at least one.",
                highlightSections: ['Workforce', 'Metrics'],
            },
        ],
        challenges: [
            { description: 'Achieve throughput above 100/hr with 15% calloff rate on all operators', evaluate: (r) => r.throughput >= 100 },
            { description: 'System survives with only 3 operators (all with calloff > 0)', evaluate: (r) => r.throughput >= 80 },
        ],
    },

    // =========================================================================
    // TIER 3: QUALITY
    // =========================================================================
    {
        id: 'cost-of-quality',
        title: 'The Cost of Quality',
        difficulty: 3,
        estimatedMinutes: 15,
        category: 'quality',
        features: ['scrap', 'rework', 'defect escape', 'inspection'],
        overview: "Scrap and rework create death spirals. Learn why the cost of quality is always higher than you think.",
        layout: {
            stations: [
                { id: 'stn-1', type: 'single', name: 'Forming', x: 230, y: 220, cycle_time: 20, cycle_time_cv: 0.1, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.05, rework_rate: 0.08, defect_detection_rate: 0.8, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
                { id: 'stn-2', type: 'single', name: 'Finishing', x: 450, y: 220, cycle_time: 25, cycle_time_cv: 0.1, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.03, rework_rate: 0.05, defect_detection_rate: 0.9, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
            ],
            sources: [
                { id: 'src-1', name: 'Material', x: 60, y: 220, arrival_distribution: 'exponential', arrival_rate: 22, arrival_cv: 0, batch_size: 1, product_types: [{ name: 'Part', ratio: 1.0, color: '#4a9f6e' }], schedule_mode: 'fixed_mix' },
            ],
            sinks: [
                { id: 'sink-1', name: 'Customer', x: 660, y: 220, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'conn-1', from_id: 'src-1', to_id: 'stn-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-2', from_id: 'stn-1', to_id: 'stn-2', buffer_capacity: 10, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-3', from_id: 'stn-2', to_id: 'sink-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 7200, speed: 50 },
        steps: [
            {
                title: 'See the Yield Cascade',
                instruction: 'Run the simulation. Forming has 5% scrap + 8% rework with 80% detection. Finishing has 3% scrap + 5% rework with 90% detection. Check the Yield and Escaped Defects metrics.',
                teaching: "Forming yields ~87% first-pass (5% scrap + 8% rework = 13% defects). But only 80% of defects are detected — the other 20% escape to Finishing. At Finishing, those escaped defects from Forming PLUS Finishing's own 8% defects combine. System yield is multiplicative: 0.87 × 0.92 ≈ 80%. One in five parts has a problem. And the ones that escape both stations reach the customer.",
                highlightElements: ['m-escaped'],
            },
            {
                title: 'The Rework Death Spiral',
                instruction: "Watch the queue at Forming carefully. Reworked parts go back into the queue and compete with new parts for capacity. What happens to WIP and lead time?",
                teaching: "Every reworked part consumes capacity twice (or more — rework can fail again). At 8% rework, you're effectively running the machine at 108% load. If it was already near capacity, the queue explodes. This is the rework death spiral: more rework → longer queues → longer lead times → more rush orders → more pressure to skip inspection → more escapes to customer.",
            },
            {
                title: 'Improve Detection Rate',
                instruction: "Increase Forming's defect detection rate to 95%. Reset and run. Compare escaped defects.",
                teaching: "Better detection catches more problems early — where they're cheap to fix. A defect caught at Forming costs one rework cycle. A defect caught at Finishing costs all previous processing plus rework. A defect that reaches the customer costs 10-100× the internal cost (warranty, returns, reputation). This is the 1-10-100 rule: fix it now, fix it later, or fail. Prevention is always cheapest.",
            },
            {
                title: 'Reduce Defect Rate at Source',
                instruction: "Reduce Forming's scrap rate to 1% and rework to 2%. Reset and run. Compare all metrics — throughput, WIP, yield, lead time.",
                teaching: "Everything improves at once. Less scrap = more parts reaching the customer. Less rework = more capacity available for new parts. Shorter queues = shorter lead times. This is why quality improvement at the source has the highest ROI of any factory investment. Deming said it: 'Quality comes not from inspection but from improvement of the production process.'",
                highlightElements: ['m-throughput', 'm-wip', 'm-leadtime'],
            },
        ],
        challenges: [
            { description: 'Achieve yield above 90%', evaluate: (r) => r.yield_rate >= 0.90 },
            { description: 'Zero escaped defects to customer', evaluate: (r) => (r.customer_returns || 0) === 0 },
        ],
    },
    {
        id: 'spc-tradeoff',
        title: 'SPC: False Alarms vs Escapes',
        difficulty: 3,
        estimatedMinutes: 15,
        category: 'quality',
        features: ['SPC', 'Western Electric', 'process drift', 'measurement lag'],
        overview: "Statistical Process Control detects drift — but every alarm stops the machine. Tight limits catch everything but cry wolf. Loose limits miss real problems.",
        layout: {
            stations: [
                { id: 'stn-1', type: 'single', name: 'CNC Mill', x: 350, y: 220, cycle_time: 25, cycle_time_cv: 0.15, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.02, rework_rate: 0.03, defect_detection_rate: 0.9, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, spc_enabled: true, spc_investigation_time: 120, drift_rate: 0.005, calibration_interval: 200, measurement_delay: 0 },
            ],
            sources: [
                { id: 'src-1', name: 'Blanks', x: 100, y: 220, arrival_distribution: 'exponential', arrival_rate: 28, arrival_cv: 0, batch_size: 1, product_types: [{ name: 'Part', ratio: 1.0, color: '#4a9f6e' }], schedule_mode: 'fixed_mix' },
            ],
            sinks: [
                { id: 'sink-1', name: 'Ship', x: 620, y: 220, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'conn-1', from_id: 'src-1', to_id: 'stn-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-2', from_id: 'stn-1', to_id: 'sink-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 14400, speed: 100 },
        steps: [
            {
                title: 'See SPC in Action',
                instruction: 'Run the simulation. The CNC Mill has SPC enabled with Western Electric rules. Watch for SPC signal events in the log — the machine stops for investigation each time.',
                teaching: "The mill has a drift rate of 0.005 — the process slowly goes out of spec between calibrations. SPC is watching: every part gets measured and checked against 4 Western Electric rules. When the chart detects a shift, the machine stops for 120s investigation. If the drift is real, it gets calibrated. If it's just random variation, it's a false alarm — 2 minutes of lost production for nothing.",
            },
            {
                title: 'Add Measurement Lag',
                instruction: "Click the CNC Mill. Set Measurement Lag to 300s (5 minutes — simulates sending samples to a CMM or lab). Reset and run.",
                teaching: "Now there's a 5-minute delay between making a part and getting the measurement result. The SPC chart is looking at data that's 5 minutes old. In those 5 minutes, the machine keeps producing parts. If a real drift started, you've made 10+ suspect parts before you even know. That's the containment scope — the blast radius of delayed information. This is why inline measurement (checking at the machine) is worth the investment.",
                highlightElements: ['m-escaped'],
            },
            {
                title: 'Tighten Investigation Time',
                instruction: "Set investigation time to 30s (fast response team). Set measurement delay back to 0. Reset and run. Compare throughput to the baseline.",
                teaching: "Faster investigation means less production time lost per signal. But the number of signals stays the same. The real lever is reducing process variation so the control chart triggers less often. That means better tooling, better fixturing, better material control — engineering solutions, not statistical ones. SPC doesn't improve quality — it tells you when quality is changing.",
            },
        ],
        challenges: [
            { description: 'Achieve throughput above 100/hr with SPC enabled', evaluate: (r) => r.throughput >= 100 },
            { description: 'Keep customer returns below 5', evaluate: (r) => (r.customer_returns || 0) < 5 },
        ],
    },

    // =========================================================================
    // TIER 4: SYSTEMS COMPLEXITY
    // =========================================================================
    {
        id: 'utility-maint-crunch',
        title: 'When Everything Breaks at Once',
        difficulty: 4,
        estimatedMinutes: 20,
        category: 'systems',
        features: ['utility failures', 'maintenance crew', 'correlated downtime'],
        overview: "Shared utilities fail and take down multiple machines simultaneously. Your maintenance crew is finite. This is where planning meets reality.",
        layout: {
            stations: [
                { id: 'stn-1', type: 'single', name: 'Press A', x: 250, y: 140, cycle_time: 22, cycle_time_cv: 0.1, changeover_time: 60, changeover_frequency: 0, uptime: 100, mtbf: 3600, mttr: 300, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.02, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
                { id: 'stn-2', type: 'single', name: 'Press B', x: 250, y: 300, cycle_time: 22, cycle_time_cv: 0.1, changeover_time: 60, changeover_frequency: 0, uptime: 100, mtbf: 3600, mttr: 300, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.02, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
                { id: 'stn-3', type: 'single', name: 'Finish', x: 500, y: 220, cycle_time: 18, cycle_time_cv: 0.1, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
            ],
            sources: [
                { id: 'src-1', name: 'Material', x: 60, y: 220, arrival_distribution: 'exponential', arrival_rate: 15, arrival_cv: 0, batch_size: 1, product_types: [{ name: 'A', ratio: 0.5, color: '#4a9f6e' }, { name: 'B', ratio: 0.5, color: '#3b82f6' }], schedule_mode: 'fixed_mix' },
            ],
            sinks: [
                { id: 'sink-1', name: 'Ship', x: 700, y: 220, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'conn-1', from_id: 'src-1', to_id: 'stn-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-2', from_id: 'src-1', to_id: 'stn-2', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-3', from_id: 'stn-1', to_id: 'stn-3', buffer_capacity: 12, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-4', from_id: 'stn-2', to_id: 'stn-3', buffer_capacity: 12, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-5', from_id: 'stn-3', to_id: 'sink-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [],
            utility_systems: [
                { id: 'util-1', name: 'Compressed Air', mtbf: 7200, mttr: 600, affected_machines: ['stn-1', 'stn-2'] },
            ],
        },
        config: { warmup: 300, runtime: 28800, speed: 200, maint_crew_size: 2 },
        steps: [
            {
                title: 'Watch the Cascade',
                instruction: 'Run the 8-hour simulation. Both presses share compressed air. When it fails, BOTH go down simultaneously. And there are only 2 maintenance techs.',
                teaching: "Correlated failures are the nightmare scenario. Individual machine failures with MTBF=3600 are manageable — one goes down, the other keeps running. But when compressed air fails, both presses stop at the same time. Now you need 2 techs just for the utility, plus any individual breakdowns that happen during. Two techs can't handle three simultaneous failures.",
                highlightSections: ['Shared Resources', 'Metrics'],
            },
            {
                title: 'Increase Maintenance Crew',
                instruction: 'Set maintenance crew to 3, then 4. Compare how throughput and maintenance queue depth change.',
                teaching: "More techs = faster recovery from correlated failures. But techs cost money. At what point does the extra tech's salary exceed the value of the production they save? This is the insurance calculation: probability × impact vs cost of prevention. Run multiple simulations and look at the variance — the average hides the bad days.",
            },
            {
                title: 'Improve Utility Reliability',
                instruction: "Click the Compressed Air utility in the sidebar. Increase MTBF to 14400 (4 hours) and reduce MTTR to 300. Reset and run.",
                teaching: "Improving the shared utility is the highest-leverage action because it prevents correlated failures. One improvement to the air system benefits both presses simultaneously. This is the systems thinking approach: fix the root cause (unreliable utility) instead of treating the symptoms (more techs to repair faster).",
            },
        ],
        challenges: [
            { description: 'Achieve throughput above 160/hr over 8 hours', evaluate: (r) => r.throughput >= 160 },
            { description: 'Keep maintenance queue under 2 on average', evaluate: (r) => true },
        ],
    },
    {
        id: 'management-chaos',
        title: 'Management by Panic',
        difficulty: 4,
        estimatedMinutes: 20,
        category: 'systems',
        features: ['management oscillation', 'policy instability', 'reactive decisions'],
        overview: "Turn on the management AI and watch it make everything worse by overreacting to metrics. The most realistic scenario in this simulator.",
        layout: {
            stations: [
                { id: 'stn-1', type: 'single', name: 'Machine A', x: 250, y: 150, cycle_time: 25, cycle_time_cv: 0.15, changeover_time: 90, changeover_frequency: 0, uptime: 100, mtbf: 5400, mttr: 300, operators: 1, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.03, rework_rate: 0.04, spc_enabled: true, spc_investigation_time: 90, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
                { id: 'stn-2', type: 'single', name: 'Machine B', x: 250, y: 300, cycle_time: 25, cycle_time_cv: 0.15, changeover_time: 90, changeover_frequency: 0, uptime: 100, mtbf: 5400, mttr: 300, operators: 1, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.03, rework_rate: 0.04, spc_enabled: true, spc_investigation_time: 90, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
                { id: 'stn-3', type: 'single', name: 'Pack', x: 500, y: 220, cycle_time: 15, cycle_time_cv: 0.1, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {} },
            ],
            sources: [
                { id: 'src-1', name: 'Orders', x: 60, y: 220, arrival_distribution: 'exponential', arrival_rate: 18, arrival_cv: 0, batch_size: 1, product_types: [{ name: 'X', ratio: 0.5, color: '#4a9f6e' }, { name: 'Y', ratio: 0.5, color: '#f39c12' }], schedule_mode: 'fixed_mix', rush_order_rate: 0.1, due_date_target: 600 },
            ],
            sinks: [
                { id: 'sink-1', name: 'Ship', x: 700, y: 220, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'conn-1', from_id: 'src-1', to_id: 'stn-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-2', from_id: 'src-1', to_id: 'stn-2', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-3', from_id: 'stn-1', to_id: 'stn-3', buffer_capacity: 15, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-4', from_id: 'stn-2', to_id: 'stn-3', buffer_capacity: 15, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-5', from_id: 'stn-3', to_id: 'sink-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [
                { id: 'op-1', name: 'Sam', skills: { 'stn-1': 0.9, 'stn-2': 0.5 }, calloffRate: 0.05, status: 'available' },
                { id: 'op-2', name: 'Jordan', skills: { 'stn-1': 0.5, 'stn-2': 0.9 }, calloffRate: 0.05, status: 'available' },
            ],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 28800, speed: 200 },
        steps: [
            {
                title: 'Establish the Baseline',
                instruction: 'Run with Management AI set to Off. Note throughput, yield, OTD, and WIP. Save this run for comparison.',
                teaching: "This factory has natural variability: breakdowns, changeovers, scrap, rework, rush orders, SPC stops. It's imperfect but stable. The system self-regulates through its queues and dispatch rules. Note the metrics — this is your best case.",
                highlightSections: ['Management AI', 'Metrics'],
            },
            {
                title: 'Turn On Nervous Management',
                instruction: "Set Management AI to 'Nervous'. Reset and run. Compare to your baseline.",
                teaching: "The nervous manager reviews metrics every hour and overreacts. WIP rising? Slash batch sizes — which means more changeovers — which means less capacity — which means MORE WIP. Yield dropping? Tighten SPC — which means more false alarms — which means more stops — which means less throughput. Every 'fix' makes something else worse. This is policy oscillation: the management equivalent of over-correcting the steering wheel.",
            },
            {
                title: 'Turn On Panicking Management',
                instruction: "Set Management AI to 'Panicking'. Reset and run. Watch the simulation log for management interventions.",
                teaching: "The panicking manager reviews every 30 minutes and takes extreme actions. They rush everything (making every order high-priority, which means no order is high-priority). They slash batch sizes to 1, then double them when WIP drops, then slash again. The system never reaches steady state. This is the real reason many factories underperform: not the equipment, not the workers — the decisions. The best improvement you can make is to stop changing things reactively and work on root causes.",
            },
        ],
        challenges: [
            { description: 'Beat the "Panicking" management throughput with Management AI off', evaluate: (r) => r.throughput >= 120 },
            { description: 'Achieve on-time delivery above 80%', evaluate: (r) => (r.on_time_delivery || 0) >= 0.80 },
        ],
    },

    // =========================================================================
    // TIER 5: MASTERY
    // =========================================================================
    {
        id: 'full-factory',
        title: 'The Full Factory Challenge',
        difficulty: 5,
        estimatedMinutes: 45,
        category: 'mastery',
        features: ['everything', 'optimization', 'tradeoffs', 'team exercise'],
        overview: "A realistic factory with every system active. Hit the targets. Every lever you pull breaks two other things. This is the capstone.",
        layout: {
            stations: [
                { id: 'stn-1', type: 'single', name: 'Cut', x: 220, y: 130, cycle_time: 18, cycle_time_cv: 0.2, changeover_time: 120, changeover_frequency: 0, uptime: 100, mtbf: 5400, mttr: 240, operators: 1, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.02, rework_rate: 0.03, defect_detection_rate: 0.85, shift_schedule: '3x8', break_duration: 1800, setup_matrix: {}, weibull_beta: 1.5, micro_stop_rate: 0.08, micro_stop_duration: 8, contamination_risk: 0.1, first_article_penalty: 0.05, first_article_count: 3, handover_loss_rate: 0.15, warmup_time: 30 },
                { id: 'stn-2', type: 'single', name: 'Form', x: 220, y: 310, cycle_time: 22, cycle_time_cv: 0.2, changeover_time: 180, changeover_frequency: 0, uptime: 100, mtbf: 4800, mttr: 360, operators: 1, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.04, rework_rate: 0.05, defect_detection_rate: 0.75, shift_schedule: '3x8', break_duration: 1800, setup_matrix: {}, weibull_beta: 2, drift_rate: 0.003, calibration_interval: 150, spc_enabled: true, spc_investigation_time: 90, micro_stop_rate: 0.1, micro_stop_duration: 12, contamination_risk: 0.15, first_article_penalty: 0.08, first_article_count: 5, handover_loss_rate: 0.2, warmup_time: 60 },
                { id: 'stn-3', type: 'single', name: 'Weld', x: 440, y: 220, cycle_time: 28, cycle_time_cv: 0.15, changeover_time: 60, changeover_frequency: 0, uptime: 100, mtbf: 7200, mttr: 300, operators: 1, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.02, rework_rate: 0.02, defect_detection_rate: 0.9, shift_schedule: '3x8', break_duration: 1800, setup_matrix: {}, micro_stop_rate: 0.05, micro_stop_duration: 6 },
                { id: 'stn-4', type: 'single', name: 'Paint', x: 640, y: 220, cycle_time: 35, cycle_time_cv: 0.1, changeover_time: 240, changeover_frequency: 0, uptime: 100, mtbf: 10800, mttr: 180, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.01, rework_rate: 0.06, defect_detection_rate: 0.95, shift_schedule: '3x8', break_duration: 1800, setup_matrix: {}, contamination_risk: 0.2, measurement_delay: 120 },
            ],
            sources: [
                { id: 'src-1', name: 'Orders', x: 40, y: 220, arrival_distribution: 'exponential', arrival_rate: 20, arrival_cv: 0, batch_size: 1, product_types: [{ name: 'Widget', ratio: 0.6, color: '#4a9f6e', shelf_life: 7200 }, { name: 'Gadget', ratio: 0.4, color: '#3b82f6', shelf_life: 10800 }], schedule_mode: 'fixed_mix', rush_order_rate: 0.12, due_date_target: 900, supplier_reliability: 0.85, incoming_quality_rate: 0.92, material_cost_per_unit: 15 },
            ],
            sinks: [
                { id: 'sink-1', name: 'Customer', x: 840, y: 220, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'conn-1', from_id: 'src-1', to_id: 'stn-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-2', from_id: 'src-1', to_id: 'stn-2', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-3', from_id: 'stn-1', to_id: 'stn-3', buffer_capacity: 8, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-4', from_id: 'stn-2', to_id: 'stn-3', buffer_capacity: 8, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-5', from_id: 'stn-3', to_id: 'stn-4', buffer_capacity: 6, transport_type: 'none', transport_distance: 0 },
                { id: 'conn-6', from_id: 'stn-4', to_id: 'sink-1', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [
                { id: 'op-1', name: 'Arun', skills: { 'stn-1': 0.9, 'stn-2': 0.3, 'stn-3': 0 }, calloffRate: 0.08, status: 'available' },
                { id: 'op-2', name: 'Priya', skills: { 'stn-1': 0.3, 'stn-2': 0.85, 'stn-3': 0.4 }, calloffRate: 0.1, status: 'available' },
                { id: 'op-3', name: 'Dev', skills: { 'stn-1': 0, 'stn-2': 0.4, 'stn-3': 0.95 }, calloffRate: 0.06, status: 'available' },
            ],
            utility_systems: [
                { id: 'util-1', name: 'Compressed Air', mtbf: 10800, mttr: 480, affected_machines: ['stn-1', 'stn-2', 'stn-3'] },
            ],
        },
        config: { warmup: 600, runtime: 28800, speed: 200, maint_crew_size: 2, revenue_per_unit: 100 },
        steps: [
            {
                title: 'Assess the Situation',
                instruction: 'Run the full 8-hour simulation. Study every metric. Identify the bottleneck, the biggest quality problem, the workforce gap, and the financial result. This is your baseline.',
                teaching: "This factory has: 4 machines with different reliability profiles, 3 operators with partial cross-training and calloff rates, 2 product types with shelf life, rush orders, supplier variability, SPC on the Form station, contamination risk, micro-stoppages, shift handover losses, startup scrap, process drift, and shared compressed air. Plus a maintenance crew of 2 and customer revenue tracking. Welcome to reality.",
            },
            {
                title: 'Find the Constraint',
                instruction: "Look at the bottleneck indicator and the utilization chart. Which machine limits throughput? Is it the machine, the operator, the tooling, or the schedule?",
                teaching: "In a system this complex, the constraint isn't always obvious. Paint has the longest cycle time (35s) but no operator requirement. Form has operator-dependent variability. Cut has high changeover. Weld might be starved because both upstream stations feed it. The constraint might shift between these depending on breakdowns, calloffs, and product mix. Your job is to find the current constraint AND the next constraint that will appear when you fix the first one.",
            },
            {
                title: 'Make Your Improvements',
                instruction: "You have freedom to change any parameter. The targets: throughput > 80/hr, on-time delivery > 75%, yield > 85%, net revenue > $0. Save runs to compare before/after.",
                teaching: "This is the real exercise. Every improvement you make affects something else. Reduce changeover time? More capacity but maybe more contamination if you rush. Add a 4th operator? More labor cost. Increase buffer sizes? More WIP, longer lead times, more ECO exposure, more expired parts. Tighten SPC? More false alarms. The solution isn't one change — it's understanding the system well enough to make the right 3-4 changes that compound positively instead of fighting each other.",
            },
            {
                title: 'Compare Your Results',
                instruction: "Use the Compare Runs tab to put your optimized run against the baseline. Where did you improve? Where did you regress? Is the net result positive?",
                teaching: "In a real factory, you'd be presenting this to management. The numbers have to add up. You can't just say 'throughput improved' — you need to show that yield didn't tank, lead times didn't double, and the P&L is positive. This is the discipline of continuous improvement: measure everything, change one thing at a time when possible, and always check for unintended consequences. Ohno stood in his circle and watched. Now you understand why.",
            },
        ],
        challenges: [
            { description: 'Throughput above 80 parts/hour', evaluate: (r) => r.throughput >= 80 },
            { description: 'On-time delivery above 75%', evaluate: (r) => (r.on_time_delivery || 0) >= 0.75 },
            { description: 'Yield above 85%', evaluate: (r) => (r.yield_rate || 0) >= 0.85 },
            { description: 'Net revenue positive', evaluate: (r) => (r.net_revenue || 0) > 0 },
        ],
    },

    // =========================================================================
    // KAIZEN METHOD SCENARIOS — each demands a specific improvement methodology
    // =========================================================================

    // --- LINE BALANCE (Foundations, Difficulty 1) ---
    {
        id: 'line-balance',
        title: 'Line Balance: Find the Rhythm',
        difficulty: 1,
        estimatedMinutes: 10,
        category: 'foundations',
        features: ['cycle time', 'takt time', 'line balance', 'starvation'],
        overview: "Three machines, wildly different cycle times. One is racing, one is crawling. Fix the imbalance before you fix anything else.",
        layout: {
            stations: [
                { id: 'lb-fast', type: 'single', name: 'Cut (Fast)', x: 250, y: 200, cycle_time: 10, cycle_time_cv: 0.05, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
                { id: 'lb-slow', type: 'single', name: 'Weld (Slow)', x: 450, y: 200, cycle_time: 45, cycle_time_cv: 0.05, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
                { id: 'lb-med', type: 'single', name: 'Pack (Medium)', x: 650, y: 200, cycle_time: 25, cycle_time_cv: 0.05, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
            ],
            sources: [
                { id: 'lb-src', name: 'Material', x: 50, y: 200, arrival_distribution: 'fixed', arrival_rate: 150, arrival_cv: 0, batch_size: 1, product_types: [{ name: 'A', ratio: 1, color: '#4fc3f7', shelf_life: 0 }], schedule_mode: 'fixed_mix', rush_order_rate: 0, due_date_target: 300, supplier_reliability: 1, incoming_quality_rate: 0, material_cost_per_unit: 0, late_delivery_penalty: 0 },
            ],
            sinks: [
                { id: 'lb-sink', name: 'Shipping', x: 850, y: 200, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'lb-c1', from_id: 'lb-src', to_id: 'lb-fast', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'lb-c2', from_id: 'lb-fast', to_id: 'lb-slow', buffer_capacity: 10, transport_type: 'none', transport_distance: 0 },
                { id: 'lb-c3', from_id: 'lb-slow', to_id: 'lb-med', buffer_capacity: 10, transport_type: 'none', transport_distance: 0 },
                { id: 'lb-c4', from_id: 'lb-med', to_id: 'lb-sink', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 3600, speed: 50 },
        steps: [
            {
                title: 'Observe the Imbalance',
                instruction: "Run the simulation and watch the buffer between Cut and Weld. What happens? Check the utilization chart after the run.",
                teaching: "Cut runs at 10s/part (360/hr capacity), Weld at 45s/part (80/hr), Pack at 25s/part (144/hr). The line can only produce at Weld's rate — 80/hr — regardless of how fast Cut and Pack are. Cut is overproducing, building WIP that Weld can never consume. This is an unbalanced line. The first step in any improvement is identifying that throughput equals the rate of the slowest station. Everything else is waste.",
                highlightElements: ['m-throughput', 'm-wip'],
            },
            {
                title: 'Balance the Line',
                instruction: "Edit the cycle times to make them as equal as possible. Target: all three machines near 20s. Run again and compare WIP and throughput.",
                teaching: "Line balancing is the practice of distributing work content equally across stations so no single station limits flow. In a real factory, this means reassigning operations between workstations — move a welding sub-operation to Cut, move a packaging sub-step to Pack. The ideal is takt time: the pace of customer demand. If a customer needs 120 parts/hour, takt time is 30s, and every station should be at or just under 30s. Perfect balance is impossible in practice, but getting close eliminates the biggest source of waste: waiting.",
                highlightElements: ['m-throughput', 'm-wip', 'm-leadtime'],
            },
            {
                title: 'Understand Takt Time',
                instruction: "The source sends 150 parts/hr (takt = 24s). Set all three cycle times to 23s. Run and observe: near-perfect flow. Now set one to 25s — just 2 seconds over takt — and watch what happens.",
                teaching: "Takt time is the heartbeat of the factory: available time / customer demand. When every station operates at or just under takt, parts flow continuously with minimal WIP. But when even one station exceeds takt by a small margin, WIP accumulates relentlessly over time. That 2-second gap doesn't look like much, but over 3600 seconds of runtime it means the line falls further and further behind. This is why line balance is the foundation — it's the first thing you fix and the last thing you stop monitoring.",
                highlightElements: ['m-throughput', 'm-wip'],
            },
        ],
        challenges: [
            { description: 'Throughput above 120 parts/hour', evaluate: (r) => r.throughput >= 120 },
            { description: 'Average WIP below 5 parts', evaluate: (r) => r.avg_wip < 5 },
        ],
    },

    // --- STANDARD WORK: TAMING VARIABILITY (Foundations, Difficulty 2) ---
    {
        id: 'standard-work',
        title: 'Standard Work: Tame the Variability',
        difficulty: 2,
        estimatedMinutes: 12,
        category: 'foundations',
        features: ['variability', 'CV', "Kingman's formula", 'standard work'],
        overview: "Two machines with identical average cycle times. One has tight standard work, the other doesn't. Watch what variability alone does to your queue.",
        layout: {
            stations: [
                { id: 'sw-a', type: 'single', name: 'Lathe (Wild)', x: 250, y: 150, cycle_time: 30, cycle_time_cv: 0.8, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
                { id: 'sw-b', type: 'single', name: 'Grind (Stable)', x: 500, y: 150, cycle_time: 30, cycle_time_cv: 0.8, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
            ],
            sources: [
                { id: 'sw-src', name: 'Raw Material', x: 50, y: 150, arrival_distribution: 'exponential', arrival_rate: 100, arrival_cv: 0.5, batch_size: 1, product_types: [{ name: 'A', ratio: 1, color: '#4fc3f7', shelf_life: 0 }], schedule_mode: 'fixed_mix', rush_order_rate: 0, due_date_target: 300, supplier_reliability: 1, incoming_quality_rate: 0, material_cost_per_unit: 0, late_delivery_penalty: 0 },
            ],
            sinks: [
                { id: 'sw-sink', name: 'Finished', x: 750, y: 150, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'sw-c1', from_id: 'sw-src', to_id: 'sw-a', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'sw-c2', from_id: 'sw-a', to_id: 'sw-b', buffer_capacity: 20, transport_type: 'none', transport_distance: 0 },
                { id: 'sw-c3', from_id: 'sw-b', to_id: 'sw-sink', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 7200, speed: 50 },
        steps: [
            {
                title: 'See Variability in Action',
                instruction: "Both machines start with CV = 0.8 (high variability). Run the simulation and note the average lead time and WIP. The average cycle time is 30s for both — plenty of capacity for 100 arrivals/hr.",
                teaching: "Kingman's formula tells us that queue length grows proportionally to the SQUARE of the coefficient of variation. CV=0.8 means some cycles take 6 seconds, others take 54. That randomness creates bunching: sometimes 3 parts arrive while the machine is on a long cycle, and the queue spikes. Standard work — documented, repeatable procedures — is the kaizen that reduces CV. It's not glamorous. It's the most important thing you can do.",
                highlightElements: ['m-wip', 'm-leadtime'],
            },
            {
                title: 'Apply Standard Work to the First Machine',
                instruction: "Select the Lathe and reduce its CV from 0.8 to 0.1. Leave the Grind unchanged at 0.8. Run again and compare lead times.",
                teaching: "Standard work means every operator performs the same task the same way every time. The cycle time doesn't have to be faster — it has to be consistent. A machine averaging 30s with CV=0.1 (28-32s range) creates smooth, predictable flow. A machine averaging 30s with CV=0.8 (6-54s range) creates chaos. Taiichi Ohno said 'Where there is no standard, there can be no kaizen.' This is what he meant — you can't improve what you can't stabilize.",
                highlightElements: ['m-leadtime', 'm-wip'],
            },
            {
                title: 'Standardize Both Machines',
                instruction: "Now set Grind's CV to 0.1 as well. Run again and compare all three runs. Notice how much lead time improved without changing cycle times at all.",
                teaching: "You didn't add capacity. You didn't add buffers. You didn't add operators. You made the process repeatable, and lead time dropped dramatically. This is the power of standard work. In a real factory, standard work documents include: the sequence of steps, the time for each step, the standard WIP required, and key quality checkpoints. They're posted at the workstation, trained into every operator, and audited regularly. Standard work is the baseline from which all other kaizen begins.",
                highlightElements: ['m-throughput', 'm-leadtime', 'm-wip'],
            },
            {
                title: 'The Arrival Variability You Cannot Control',
                instruction: "Note that arrival CV is 0.5 (customer orders are inherently variable). Even with perfect standard work (CV=0.1 on both machines), some WIP remains. Why?",
                teaching: "You can standardize your process, but you cannot standardize your customer. Arrival variability is external — it's the pattern of customer orders. Kingman's formula uses BOTH arrival CV and process CV. Standard work drives process CV down, but the arrival term remains. This is why buffers exist even in well-run factories: they absorb the variability you cannot eliminate. The goal isn't zero WIP — it's the minimum WIP needed to absorb external variability while maintaining flow. Toyota calls this 'standard WIP' — the inventory the system needs to function.",
                highlightElements: ['m-wip', 'm-leadtime'],
            },
        ],
        challenges: [
            { description: 'Average lead time below 90 seconds', evaluate: (r) => r.avg_lead_time < 90 },
            { description: 'Average WIP below 4 parts', evaluate: (r) => r.avg_wip < 4 },
        ],
    },

    // --- SMED: CUT THE CHANGEOVER (Variability, Difficulty 2) ---
    {
        id: 'smed-kaizen',
        title: 'SMED: Cut the Changeover',
        difficulty: 2,
        estimatedMinutes: 15,
        category: 'variability',
        features: ['SMED', 'changeover', 'batch size', 'one-piece flow'],
        overview: "A press with a 10-minute changeover running two product types. Shigeo Shingo showed us how to get that under a minute. Your turn.",
        layout: {
            stations: [
                { id: 'sm-press', type: 'single', name: 'Press', x: 300, y: 200, cycle_time: 20, cycle_time_cv: 0.1, changeover_time: 600, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.02, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
                { id: 'sm-finish', type: 'single', name: 'Finish', x: 550, y: 200, cycle_time: 18, cycle_time_cv: 0.1, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
            ],
            sources: [
                { id: 'sm-src', name: 'Orders', x: 50, y: 200, arrival_distribution: 'exponential', arrival_rate: 120, arrival_cv: 0.3, batch_size: 1, product_types: [{ name: 'Widget', ratio: 0.5, color: '#4fc3f7', shelf_life: 0 }, { name: 'Bracket', ratio: 0.5, color: '#ff8a65', shelf_life: 0 }], schedule_mode: 'fixed_mix', rush_order_rate: 0.1, due_date_target: 300, supplier_reliability: 1, incoming_quality_rate: 0, material_cost_per_unit: 0, late_delivery_penalty: 0 },
            ],
            sinks: [
                { id: 'sm-sink', name: 'Shipping', x: 800, y: 200, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'sm-c1', from_id: 'sm-src', to_id: 'sm-press', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'sm-c2', from_id: 'sm-press', to_id: 'sm-finish', buffer_capacity: 15, transport_type: 'none', transport_distance: 0 },
                { id: 'sm-c3', from_id: 'sm-finish', to_id: 'sm-sink', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 7200, speed: 50 },
        steps: [
            {
                title: 'Feel the Pain of a Long Changeover',
                instruction: "Run the baseline. The Press has a 600s (10 min) changeover every time the product type switches. Watch the utilization chart — how much time does the Press spend in setup vs processing?",
                teaching: "With 50/50 product mix and random arrivals, the Press changes over frequently. Each 10-minute changeover is 10 minutes of zero output while the queue grows. The traditional response is to run large batches — make all Widget orders first, then switch to Bracket. This minimizes changeovers but creates lead time problems: Bracket orders wait while you're running the Widget batch. Shigeo Shingo's insight was simple: don't accept the changeover time — reduce it.",
                highlightElements: ['m-throughput', 'm-leadtime'],
            },
            {
                title: 'Apply SMED — Separate Internal and External',
                instruction: "SMED Stage 1: Separate internal setup (machine stopped) from external (done while running). Reduce changeover from 600s to 180s. Run again.",
                teaching: "Shingo observed that much of changeover time was spent on tasks that could be done while the machine was still running the previous batch: staging tools, preheating dies, preparing materials, filling out paperwork. By separating internal setup (must be done with machine stopped) from external setup (can be done while running), you typically cut changeover 30-50% with no investment — just organization. The 600→180s reduction represents moving die staging, parameter lookup, and material prep to external time.",
                highlightElements: ['m-throughput', 'm-leadtime'],
            },
            {
                title: 'SMED Stage 2 — Convert and Streamline',
                instruction: "Reduce changeover to 60s (quick-release clamps, standardized die heights, one-turn fasteners). Run again. Now try a really aggressive target: 20s.",
                teaching: "SMED Stage 2 converts remaining internal operations to external through engineering: quick-release clamps replace bolts, standardized die heights eliminate shimming, intermediate jigs allow offline setup. Stage 3 streamlines everything remaining: parallel operations (two people), eliminate adjustments through precision, eliminate trial runs through mistake-proofing. Shingo's original target was under 10 minutes (Single Minute Exchange of Die), but world-class operations achieve under 1 minute. A Formula 1 pit stop is SMED perfected: what once took minutes now takes seconds.",
                highlightElements: ['m-throughput', 'm-leadtime'],
            },
            {
                title: 'The Real Prize: Smaller Batches',
                instruction: "With changeover at 60s, lead time dropped. But the real prize is flexibility. Notice that with fast changeovers, you can mix product types freely without throughput penalty. Check on-time delivery — rush orders get served faster.",
                teaching: "The point of SMED is NOT just to recover changeover time as capacity (though that helps). The real benefit is the freedom to run smaller batches. Small batches mean shorter lead times, less WIP, faster response to demand changes, and fewer parts at risk from ECOs or quality escapes. When changeover is 10 minutes, you batch to amortize the cost. When changeover is 60 seconds, you can change over every part if needed. That's one-piece flow — Ohno's ideal. SMED doesn't just save time; it changes your entire production strategy.",
                highlightElements: ['m-leadtime', 'm-wip'],
            },
        ],
        challenges: [
            { description: 'Throughput above 100 parts/hour', evaluate: (r) => r.throughput >= 100 },
            { description: 'Average lead time below 120 seconds', evaluate: (r) => r.avg_lead_time < 120 },
        ],
    },

    // --- POKA-YOKE: ERROR-PROOF THE PROCESS (Quality, Difficulty 2) ---
    {
        id: 'poka-yoke',
        title: 'Poka-Yoke: Error-Proof the Process',
        difficulty: 2,
        estimatedMinutes: 12,
        category: 'quality',
        features: ['poka-yoke', 'detection rate', 'escaped defects', 'cost of quality'],
        overview: "A machine makes defects and nobody catches them until the customer does. Inspection helps. Preventing the defect at the source is better.",
        layout: {
            stations: [
                { id: 'pk-mach', type: 'single', name: 'Assembly', x: 300, y: 200, cycle_time: 25, cycle_time_cv: 0.15, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.10, rework_rate: 0.05, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 0.5 },
            ],
            sources: [
                { id: 'pk-src', name: 'Parts', x: 50, y: 200, arrival_distribution: 'fixed', arrival_rate: 120, arrival_cv: 0, batch_size: 1, product_types: [{ name: 'A', ratio: 1, color: '#4fc3f7', shelf_life: 0 }], schedule_mode: 'fixed_mix', rush_order_rate: 0, due_date_target: 300, supplier_reliability: 1, incoming_quality_rate: 0, material_cost_per_unit: 0, late_delivery_penalty: 0 },
            ],
            sinks: [
                { id: 'pk-sink', name: 'Customer', x: 600, y: 200, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'pk-c1', from_id: 'pk-src', to_id: 'pk-mach', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'pk-c2', from_id: 'pk-mach', to_id: 'pk-sink', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 7200, speed: 50, revenue_per_unit: 50 },
        steps: [
            {
                title: 'See the Escapes',
                instruction: "Run the baseline. Assembly has 10% scrap, 5% rework, but only 50% detection rate. That means half the defects reach the customer. Check escaped defects and customer returns in the results.",
                teaching: "Detection rate is the probability that a defective part gets caught before shipping. At 50%, you're essentially flipping a coin on every defect. With 15% total defect rate (10% scrap + 5% rework) and 50% detection, roughly 7.5% of all parts reach the customer as defects. That's a customer return rate that kills businesses. The traditional response is 'add more inspection.' That's necessary but insufficient.",
                highlightElements: ['m-throughput'],
            },
            {
                title: 'Add Inspection (Detection Poka-Yoke)',
                instruction: "Increase the defect detection rate to 0.95 (95%). This represents adding a go/no-go gauge, a visual check fixture, or an automated sensor. Run again.",
                teaching: "Detection poka-yoke catches defects before they leave the station: go/no-go gauges, proximity sensors, weight checks, vision systems, shape-matching fixtures. At 95% detection, escaped defects drop dramatically. But notice: you're still making defects — you're just catching them. Every caught defect is a scrapped or reworked part. You're spending capacity making bad parts and then spending more capacity catching them. Inspection is a net that prevents customer damage, but it's not improvement — it's damage control.",
                highlightElements: ['m-throughput'],
            },
            {
                title: 'Prevent the Defect at Source (Prevention Poka-Yoke)',
                instruction: "Now reduce scrap rate from 0.10 to 0.02 and rework rate from 0.05 to 0.01. This represents error-proofing the process itself: asymmetric fixtures (parts can only go in one way), automatic torque control, material sensors. Run again and compare net revenue across all three runs.",
                teaching: "Prevention poka-yoke makes the error impossible or immediately obvious: asymmetric connectors that only mate one way, bin systems with counted kits, interlocking fixtures, limit switches that stop the machine if a step is missed. Shingo distinguished three levels: (1) Setting poka-yoke — prevent the error condition from occurring; (2) Contact poka-yoke — physical shape or property prevents wrong assembly; (3) Motion-step poka-yoke — enforce the correct sequence. Prevention > detection > reaction. Always. The 1-10-100 rule: $1 to prevent, $10 to detect, $100 to fix after shipping.",
                highlightElements: ['m-throughput'],
            },
        ],
        challenges: [
            { description: 'Customer returns below 5', evaluate: (r) => (r.customer_returns || 0) < 5 },
            { description: 'Yield above 95%', evaluate: (r) => (r.yield_rate || 0) >= 0.95 },
        ],
    },

    // --- TPM: OWN YOUR MACHINE (Variability, Difficulty 3) ---
    {
        id: 'tpm-journey',
        title: 'TPM: Own Your Machine',
        difficulty: 3,
        estimatedMinutes: 18,
        category: 'variability',
        features: ['TPM', 'MTBF', 'MTTR', 'Weibull', 'autonomous maintenance'],
        overview: "Two machines breaking down constantly. Maintenance is overwhelmed. TPM says operators should own their machines — clean, inspect, lubricate, and detect problems before they become failures.",
        layout: {
            stations: [
                { id: 'tpm-a', type: 'single', name: 'Lathe', x: 250, y: 150, cycle_time: 25, cycle_time_cv: 0.15, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: 1800, mttr: 600, operators: 1, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.03, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0.05, micro_stop_duration: 5, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
                { id: 'tpm-b', type: 'single', name: 'Mill', x: 500, y: 150, cycle_time: 28, cycle_time_cv: 0.15, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: 2400, mttr: 900, operators: 1, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.03, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0.08, micro_stop_duration: 8, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
            ],
            sources: [
                { id: 'tpm-src', name: 'Raw Stock', x: 50, y: 150, arrival_distribution: 'exponential', arrival_rate: 100, arrival_cv: 0.3, batch_size: 1, product_types: [{ name: 'Shaft', ratio: 1, color: '#81c784', shelf_life: 0 }], schedule_mode: 'fixed_mix', rush_order_rate: 0, due_date_target: 600, supplier_reliability: 1, incoming_quality_rate: 0, material_cost_per_unit: 0, late_delivery_penalty: 0 },
            ],
            sinks: [
                { id: 'tpm-sink', name: 'QC Out', x: 750, y: 150, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'tpm-c1', from_id: 'tpm-src', to_id: 'tpm-a', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'tpm-c2', from_id: 'tpm-a', to_id: 'tpm-b', buffer_capacity: 10, transport_type: 'none', transport_distance: 0 },
                { id: 'tpm-c3', from_id: 'tpm-b', to_id: 'tpm-sink', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [
                { id: 'tpm-op1', name: 'Machinist A', skills: { 'tpm-a': 0.8, 'tpm-b': 0.3 }, calloffRate: 0, status: 'available' },
                { id: 'tpm-op2', name: 'Machinist B', skills: { 'tpm-a': 0.3, 'tpm-b': 0.8 }, calloffRate: 0, status: 'available' },
            ],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 14400, speed: 100, maint_crew_size: 1 },
        steps: [
            {
                title: 'Observe the Breakdown Pattern',
                instruction: "Run the baseline. Both machines have random failures (Weibull β=1, exponential). MTBF 1800s/2400s, MTTR 600s/900s. Note micro-stoppages too (5-8%). Check downtime percentage and throughput.",
                teaching: "Weibull β=1 means failures are random — there's no pattern, no warning. The machine is equally likely to fail in the next second whether it ran for 10 minutes or 10 hours. With a single maintenance crew, when both machines break simultaneously, one waits while the other gets fixed. Micro-stoppages (brief jams, sensor trips) add hidden downtime that doesn't show as 'breakdown' but eats capacity. This is the state before TPM: reactive maintenance, firefighting, and chronic capacity loss.",
                highlightElements: ['m-throughput', 'm-wip'],
            },
            {
                title: 'Step 1 — Autonomous Maintenance (Reduce Micro-Stops)',
                instruction: "TPM starts with operators cleaning, inspecting, and lubricating their own machines. Reduce micro_stop_rate to 0.01 on both machines (operators catch jams before they happen). Run again.",
                teaching: "The first pillar of TPM is autonomous maintenance — Jishu Hozen. Operators perform daily cleaning, inspection, and lubrication. This does three things: (1) Eliminates minor stoppages caused by dirt, chip buildup, loose fasteners; (2) Develops operator awareness — they notice when something sounds different, feels different, vibrates differently; (3) Frees maintenance technicians to focus on major repairs and improvement. Nakajima's 7 steps of autonomous maintenance start with initial cleaning and end with operators who truly own their machines.",
                highlightElements: ['m-throughput'],
            },
            {
                title: 'Step 2 — Planned Maintenance (Change Failure Pattern)',
                instruction: "With regular PM, failures shift from random to wear-out. Change Weibull β from 1 to 3 on both machines. This means failures cluster around a predictable age. Also increase MTBF to 3600s (Lathe) and 4800s (Mill) — PM catches problems before they become failures. Run again.",
                teaching: "Weibull β>2 means failures follow a wear-out pattern — the machine is more likely to fail after a certain age. This is actually GOOD, because you can now predict and schedule maintenance. You replace bearings at 3000 hours because you know they fail around 3500. You change hydraulic seals on a calendar because you know the degradation curve. Planned maintenance replaces reactive firefighting with scheduled prevention. MTBF increases because you're catching and fixing degradation before it becomes a failure event.",
                highlightElements: ['m-throughput', 'm-wip'],
            },
            {
                title: 'Step 3 — Reduce MTTR Through Cross-Training',
                instruction: "Cross-train both operators on both machines (set all skills to 0.7+). Reduce MTTR to 300s (Lathe) and 400s (Mill) — faster diagnosis because operators understand the machine. Run the final comparison.",
                teaching: "When operators understand their machines deeply, they diagnose faster. 'It started making this noise yesterday' is worth 30 minutes of troubleshooting. Cross-training also means when one machine is down, its operator can help on the other instead of standing idle. The TPM vision is zero breakdowns, zero defects, zero accidents. You won't reach zero, but the journey from reactive to predictive transforms throughput, quality, and morale. Operators who own their machines take pride in uptime. That cultural shift is worth more than any maintenance schedule.",
                highlightElements: ['m-throughput', 'm-wip', 'm-leadtime'],
            },
        ],
        challenges: [
            { description: 'Throughput above 90 parts/hour', evaluate: (r) => r.throughput >= 90 },
            { description: 'Total downtime below 10% on both machines', evaluate: (r) => {
                for (const [id, s] of Object.entries(r.station_utilizations || {})) {
                    if ((s.down || 0) > 0.10) return false;
                }
                return true;
            }},
        ],
    },

    // --- JIDOKA: STOP AND FIX (Quality, Difficulty 3) ---
    {
        id: 'jidoka',
        title: 'Jidoka: Stop and Fix',
        difficulty: 3,
        estimatedMinutes: 15,
        category: 'quality',
        features: ['jidoka', 'SPC', 'process drift', 'quality at source'],
        overview: "A process drifts out of spec and nobody notices until 200 defective parts reach the customer. Jidoka says: detect the abnormality, stop, fix, then resume. Short-term pain, long-term gain.",
        layout: {
            stations: [
                { id: 'jd-form', type: 'single', name: 'Forming', x: 250, y: 200, cycle_time: 22, cycle_time_cv: 0.1, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.02, rework_rate: 0.01, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0.008, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 0.7 },
                { id: 'jd-finish', type: 'single', name: 'Finishing', x: 500, y: 200, cycle_time: 20, cycle_time_cv: 0.1, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
            ],
            sources: [
                { id: 'jd-src', name: 'Blanks', x: 50, y: 200, arrival_distribution: 'fixed', arrival_rate: 130, arrival_cv: 0, batch_size: 1, product_types: [{ name: 'Part', ratio: 1, color: '#ce93d8', shelf_life: 0 }], schedule_mode: 'fixed_mix', rush_order_rate: 0, due_date_target: 300, supplier_reliability: 1, incoming_quality_rate: 0, material_cost_per_unit: 0, late_delivery_penalty: 0 },
            ],
            sinks: [
                { id: 'jd-sink', name: 'Customer', x: 750, y: 200, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'jd-c1', from_id: 'jd-src', to_id: 'jd-form', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'jd-c2', from_id: 'jd-form', to_id: 'jd-finish', buffer_capacity: 10, transport_type: 'none', transport_distance: 0 },
                { id: 'jd-c3', from_id: 'jd-finish', to_id: 'jd-sink', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 14400, speed: 100, revenue_per_unit: 75 },
        steps: [
            {
                title: 'Watch the Drift',
                instruction: "Run the baseline. Forming has a drift rate of 0.008 — the process slowly moves off-center. With SPC disabled and only 70% detection, defects escape to the customer over time. Check customer returns and net revenue.",
                teaching: "Process drift is the slow, insidious degradation that operators don't notice because it happens gradually: tool wear, thermal expansion, material variation accumulation, fixture loosening. Without statistical detection, the process drifts until defects become obvious — by which point hundreds of parts have shipped. Sakichi Toyoda's original jidoka concept was a loom that stopped automatically when a thread broke. The principle: build quality into the process by detecting abnormalities and stopping immediately.",
                highlightElements: ['m-throughput'],
            },
            {
                title: 'Enable Jidoka (SPC Auto-Stop)',
                instruction: "Select Forming and enable SPC. Set investigation time to 120s (time to diagnose and correct the drift). Run again. Throughput will drop — but watch what happens to customer returns.",
                teaching: "Enabling SPC is enabling jidoka: the machine now monitors itself using Western Electric rules and stops when it detects a signal — a point beyond control limits, a run of 8 above the mean, increasing trends. The 120s investigation time is the cost of stopping. Management hates it: 'You're stopping the machine! We're losing output!' Yes. But every minute the drifted process runs, it produces parts that will come back as returns at 3× the revenue. The throughput loss from stopping is visible and immediate. The cost of NOT stopping is delayed, distributed, and devastating.",
                highlightElements: ['m-throughput'],
            },
            {
                title: 'Optimize the Response',
                instruction: "Reduce investigation time to 60s (better-trained operators with pre-positioned tools and clear standard work for SPC response). Also increase detection rate to 0.9. Run again.",
                teaching: "Jidoka is not just 'stop when broken.' It's a system: (1) Detect the abnormality (SPC charts, andon lights, sensors); (2) Stop (automatically or by operator pull); (3) Fix the immediate problem (adjust, recalibrate, replace tool); (4) Investigate root cause (why did it drift? Can we prevent recurrence?). Reducing investigation time isn't about rushing — it's about preparation. Standard work for SPC response: the corrective actions are documented, tools are staged, replacement parts are kitted. When the signal fires, the operator executes a practiced routine, not a panicked improvisation.",
                highlightElements: ['m-throughput'],
            },
            {
                title: 'Compare the Economics',
                instruction: "Compare all three runs. Throughput with SPC is lower than without. But net revenue tells the real story. The 'slower' line with jidoka is more profitable because it doesn't hemorrhage money on returns.",
                teaching: "This is the essential jidoka trade-off: throughput vs quality. A manager looking only at parts/hour will shut off the SPC and celebrate the throughput increase — until the warranty claims arrive. Deming understood this: 'You can not inspect quality into a product.' Jidoka builds quality into the process by refusing to pass defects forward. Toyota's andon cord philosophy — any worker can stop the line — seems wasteful until you calculate the cost of NOT stopping. The fastest line is the one that never makes a defect.",
                highlightElements: ['m-throughput'],
            },
        ],
        challenges: [
            { description: 'Customer returns below 10', evaluate: (r) => (r.customer_returns || 0) < 10 },
            { description: 'Net revenue positive', evaluate: (r) => (r.net_revenue || 0) > 0 },
        ],
    },

    // --- HEIJUNKA: LEVEL THE LOAD (Systems, Difficulty 3) ---
    {
        id: 'heijunka',
        title: 'Heijunka: Level the Load',
        difficulty: 3,
        estimatedMinutes: 15,
        category: 'systems',
        features: ['heijunka', 'production leveling', 'batch size', 'WIP smoothing'],
        overview: "Big batches of one product, then big batches of another. WIP explodes, lead times spike, and the downstream stations alternate between starved and flooded. Level the production.",
        layout: {
            stations: [
                { id: 'hj-press', type: 'single', name: 'Press', x: 250, y: 200, cycle_time: 20, cycle_time_cv: 0.1, changeover_time: 120, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.02, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
                { id: 'hj-coat', type: 'single', name: 'Coat', x: 450, y: 200, cycle_time: 22, cycle_time_cv: 0.1, changeover_time: 60, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
                { id: 'hj-pack', type: 'single', name: 'Pack', x: 650, y: 200, cycle_time: 18, cycle_time_cv: 0.05, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
            ],
            sources: [
                { id: 'hj-src', name: 'Demand', x: 50, y: 200, arrival_distribution: 'exponential', arrival_rate: 100, arrival_cv: 0.4, batch_size: 5, product_types: [{ name: 'Red', ratio: 0.6, color: '#ef5350', shelf_life: 0 }, { name: 'Blue', ratio: 0.4, color: '#42a5f5', shelf_life: 0 }], schedule_mode: 'batch_sequence', rush_order_rate: 0.05, due_date_target: 600, supplier_reliability: 1, incoming_quality_rate: 0, material_cost_per_unit: 0, late_delivery_penalty: 0 },
            ],
            sinks: [
                { id: 'hj-sink', name: 'Shipping', x: 850, y: 200, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'hj-c1', from_id: 'hj-src', to_id: 'hj-press', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'hj-c2', from_id: 'hj-press', to_id: 'hj-coat', buffer_capacity: 15, transport_type: 'none', transport_distance: 0 },
                { id: 'hj-c3', from_id: 'hj-coat', to_id: 'hj-pack', buffer_capacity: 15, transport_type: 'none', transport_distance: 0 },
                { id: 'hj-c4', from_id: 'hj-pack', to_id: 'hj-sink', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 14400, speed: 100 },
        steps: [
            {
                title: 'See the Batching Problem',
                instruction: "Run the baseline. Demand arrives in batches of 5 using batch_sequence mode (all Red, then all Blue). Watch the WIP history — it oscillates wildly. Check average lead time.",
                teaching: "Batch-and-queue is the default mode of most factories: accumulate orders, run a big batch of Type A, changeover, run a big batch of Type B. It feels efficient because changeover time is 'amortized' over more parts. But it creates mura (unevenness): downstream stations get flooded with one product type, then starved, then flooded with the other. WIP oscillates, lead times spike for the product type that's waiting, and the whole system amplifies any variability in demand. The batch arrived in groups of 5 — by the time it passes through 3 stations, the unevenness is magnified.",
                highlightElements: ['m-wip', 'm-leadtime'],
            },
            {
                title: 'Level the Volume',
                instruction: "Change the source batch_size to 1 (one part at a time) but keep batch_sequence mode. Run again. WIP should be smoother, but changeovers increase.",
                teaching: "Heijunka (production leveling) has two dimensions: volume and mix. Leveling volume means releasing work at a steady rate instead of in lumps. Even if the customer orders in batches of 50, the heijunka box breaks that into small, evenly-spaced releases. This is the pacemaker concept: the upstream processes should receive work at a consistent rhythm, regardless of how the customer orders arrived. Smaller release batches create smoother flow through the entire system.",
                highlightElements: ['m-wip', 'm-leadtime'],
            },
            {
                title: 'Level the Mix',
                instruction: "Change schedule_mode from 'batch_sequence' to 'fixed_mix'. Now Red and Blue arrive interleaved according to their ratio (60/40). Run again.",
                teaching: "Leveling the mix means instead of RRRRR-BBBBB (5 Red, then 5 Blue), you produce R-R-B-R-R-B (interleaved by ratio). This is Toyota's heijunka box: every time slot has a predetermined product type. The changeovers increase, but each changeover is small and predictable. Every product type flows through the system continuously, so no product waits for 'its batch' to be scheduled. This only works when changeover times are short (see SMED scenario). Heijunka and SMED are complementary: SMED enables small batches, heijunka ensures you use that capability.",
                highlightElements: ['m-wip', 'm-leadtime'],
            },
            {
                title: 'Reduce Changeover to Enable Flow',
                instruction: "Reduce Press changeover to 30s and Coat changeover to 15s (the SMED improvements). Run again with fixed_mix mode. Compare lead time and on-time delivery across all runs.",
                teaching: "With fast changeovers and leveled mix, the system approaches one-piece flow: every part moves individually through every station without batching delay. Lead time drops to near the sum of cycle times. WIP stays low and stable. On-time delivery improves because no product type is 'waiting its turn.' This is the Toyota Production System ideal: make what the customer wants, when they want it, in the quantity they want, with minimum waste. Heijunka is the mechanism that makes this possible at scale.",
                highlightElements: ['m-throughput', 'm-leadtime', 'm-wip'],
            },
        ],
        challenges: [
            { description: 'Average lead time below 150 seconds', evaluate: (r) => r.avg_lead_time < 150 },
            { description: 'Average WIP below 6 parts', evaluate: (r) => r.avg_wip < 6 },
        ],
    },

    // --- THE KAIZEN BLITZ (Systems, Difficulty 4) ---
    {
        id: 'kaizen-blitz',
        title: 'The Kaizen Blitz',
        difficulty: 4,
        estimatedMinutes: 25,
        category: 'systems',
        features: ['PDCA', 'kaizen event', 'constraint ID', 'iterative improvement'],
        overview: "A messy 4-machine line with multiple problems: variability, breakdowns, quality issues, and long changeovers. You have one kaizen event. Find the biggest loss, fix it, measure, repeat. This is PDCA.",
        layout: {
            stations: [
                { id: 'kb-cut', type: 'single', name: 'Cut', x: 200, y: 200, cycle_time: 20, cycle_time_cv: 0.5, changeover_time: 180, changeover_frequency: 0, uptime: 100, mtbf: 3600, mttr: 300, operators: 1, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.05, rework_rate: 0.03, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1.5, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0.03, micro_stop_duration: 5, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 0.8 },
                { id: 'kb-weld', type: 'single', name: 'Weld', x: 380, y: 200, cycle_time: 30, cycle_time_cv: 0.4, changeover_time: 60, changeover_frequency: 0, uptime: 100, mtbf: 5400, mttr: 600, operators: 1, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.08, rework_rate: 0.04, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0.003, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0.04, micro_stop_duration: 8, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 0.7 },
                { id: 'kb-grind', type: 'single', name: 'Grind', x: 560, y: 200, cycle_time: 25, cycle_time_cv: 0.3, changeover_time: 30, changeover_frequency: 0, uptime: 100, mtbf: 7200, mttr: 200, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.02, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 2, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
                { id: 'kb-pack', type: 'single', name: 'Pack', x: 740, y: 200, cycle_time: 15, cycle_time_cv: 0.15, changeover_time: 0, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 1, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
            ],
            sources: [
                { id: 'kb-src', name: 'Orders', x: 30, y: 200, arrival_distribution: 'exponential', arrival_rate: 100, arrival_cv: 0.4, batch_size: 1, product_types: [{ name: 'Standard', ratio: 0.7, color: '#4fc3f7', shelf_life: 0 }, { name: 'Premium', ratio: 0.3, color: '#ffd54f', shelf_life: 0 }], schedule_mode: 'fixed_mix', rush_order_rate: 0.1, due_date_target: 600, supplier_reliability: 1, incoming_quality_rate: 0, material_cost_per_unit: 0, late_delivery_penalty: 0 },
            ],
            sinks: [
                { id: 'kb-sink', name: 'Shipping', x: 920, y: 200, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'kb-c1', from_id: 'kb-src', to_id: 'kb-cut', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
                { id: 'kb-c2', from_id: 'kb-cut', to_id: 'kb-weld', buffer_capacity: 10, transport_type: 'none', transport_distance: 0 },
                { id: 'kb-c3', from_id: 'kb-weld', to_id: 'kb-grind', buffer_capacity: 10, transport_type: 'none', transport_distance: 0 },
                { id: 'kb-c4', from_id: 'kb-grind', to_id: 'kb-pack', buffer_capacity: 10, transport_type: 'none', transport_distance: 0 },
                { id: 'kb-c5', from_id: 'kb-pack', to_id: 'kb-sink', buffer_capacity: null, transport_type: 'none', transport_distance: 0 },
            ],
            work_centers: [],
            operators: [
                { id: 'kb-op1', name: 'Ravi', skills: { 'kb-cut': 0.9, 'kb-weld': 0.4, 'kb-pack': 0.6 }, calloffRate: 0.05, status: 'available' },
                { id: 'kb-op2', name: 'Meera', skills: { 'kb-cut': 0.4, 'kb-weld': 0.9, 'kb-pack': 0.7 }, calloffRate: 0.05, status: 'available' },
                { id: 'kb-op3', name: 'Suresh', skills: { 'kb-cut': 0.6, 'kb-weld': 0.6, 'kb-pack': 0.9 }, calloffRate: 0.05, status: 'available' },
            ],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 14400, speed: 100, revenue_per_unit: 80 },
        steps: [
            {
                title: 'Plan — Measure the Baseline',
                instruction: "Run the simulation without changes. Save this run. Look at the utilization chart, quality metrics, throughput, and lead time. Write down (mentally) the three biggest losses you can identify.",
                teaching: "A kaizen blitz (or kaizen event) is a focused 3-5 day improvement sprint. Day 1 is always measurement: you go to gemba (the actual place), observe the actual process, and collect data on the actual product. No opinions. No assumptions. Just measurement. The PDCA cycle starts with Plan: understand the current state before proposing any changes. The biggest mistake in kaizen events is jumping to solutions. Resist the urge. The data will tell you where to focus.",
                highlightElements: ['m-throughput', 'm-wip', 'm-leadtime'],
            },
            {
                title: 'Do — Fix the Biggest Constraint',
                instruction: "Identify which machine is the bottleneck (highest utilization, longest queue). It should be Weld: 30s cycle time, 0.4 CV, breakdowns, 8% scrap, 4% rework, drift. Fix its BIGGEST problem — the one that would recover the most capacity. Save this run.",
                teaching: "Goldratt's 5 Focusing Steps: (1) Identify the constraint, (2) Exploit it (maximize its output), (3) Subordinate everything else to it, (4) Elevate it (invest to increase capacity), (5) Go back to step 1. In a kaizen blitz, you identify the constraint and then ask: what is the single biggest loss on this machine? Is it downtime (fix reliability)? Scrap (fix quality)? Variability (standardize)? Setup (SMED)? Fix the biggest one first. Don't scatter your effort across all problems simultaneously — that's management by committee, not kaizen.",
                highlightElements: ['m-throughput', 'm-wip'],
            },
            {
                title: 'Check — Measure the Impact',
                instruction: "Compare your improved run against the baseline using the Compare Runs tab. Did throughput improve? Did a new bottleneck emerge? Identify the next biggest loss.",
                teaching: "Check is the discipline that separates kaizen from tinkering. You made a specific change based on a specific hypothesis ('reducing Weld scrap will increase effective capacity'). Now verify: did throughput actually increase? By how much? Did anything else get worse? If the bottleneck shifted to another machine, that's success — you elevated the constraint. If it didn't shift, you either fixed the wrong problem or didn't fix it enough. This is the scientific method applied to manufacturing: hypothesis → experiment → analysis → conclusion.",
                highlightElements: ['m-throughput', 'm-wip', 'm-leadtime'],
            },
            {
                title: 'Act — Iterate and Standardize',
                instruction: "Fix the next biggest problem. Then the next. After 3 improvement cycles, save and compare against the original baseline. Target: throughput > 75/hr, yield > 90%, on-time delivery > 70%.",
                teaching: "Act means two things: (1) If the improvement worked, standardize it — document the new settings, train the operators, update the standard work. (2) Start the next PDCA cycle. Kaizen is not a one-time event. It's an infinite loop: Plan → Do → Check → Act → Plan → Do → Check → Act. Each cycle makes the system a little better. The kaizen blitz format compresses this into days, but the mindset operates every day. Masaaki Imai said: 'The message of kaizen is that not a day should go by without some kind of improvement being made somewhere in the company.'",
                highlightElements: ['m-throughput', 'm-wip', 'm-leadtime'],
            },
        ],
        challenges: [
            { description: 'Throughput above 75 parts/hour', evaluate: (r) => r.throughput >= 75 },
            { description: 'Yield above 90%', evaluate: (r) => (r.yield_rate || 0) >= 0.90 },
            { description: 'On-time delivery above 70%', evaluate: (r) => (r.on_time_delivery || 0) >= 0.70 },
        ],
    },

    // --- VALUE STREAM MAPPING: SEE THE WASTE (Systems, Difficulty 3) ---
    {
        id: 'value-stream',
        title: 'Value Stream: See the Waste',
        difficulty: 3,
        estimatedMinutes: 18,
        category: 'systems',
        features: ['value stream', 'lead time', '7 wastes', 'value-add ratio'],
        overview: "A 4-machine line where parts spend 95% of their time waiting. The processing is fine. The system is the problem. Map the value stream and eliminate the waste.",
        layout: {
            stations: [
                { id: 'vs-cut', type: 'single', name: 'Cut', x: 180, y: 200, cycle_time: 15, cycle_time_cv: 0.2, changeover_time: 120, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.01, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
                { id: 'vs-drill', type: 'single', name: 'Drill', x: 370, y: 200, cycle_time: 12, cycle_time_cv: 0.15, changeover_time: 90, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.01, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
                { id: 'vs-bend', type: 'single', name: 'Bend', x: 560, y: 200, cycle_time: 18, cycle_time_cv: 0.25, changeover_time: 60, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.02, rework_rate: 0, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
                { id: 'vs-assy', type: 'single', name: 'Assembly', x: 750, y: 200, cycle_time: 20, cycle_time_cv: 0.3, changeover_time: 30, changeover_frequency: 0, uptime: 100, mtbf: null, mttr: null, operators: 0, batch_size: 1, setup_time: 0, work_center_id: null, scrap_rate: 0.01, rework_rate: 0.02, shift_schedule: '24/7', break_duration: 0, setup_matrix: {}, weibull_beta: 1, drift_rate: 0, calibration_interval: 0, spc_enabled: false, spc_investigation_time: 0, micro_stop_rate: 0, micro_stop_duration: 0, contamination_risk: 0, first_article_penalty: 0, first_article_count: 0, handover_loss_rate: 0, warmup_time: 0, measurement_delay: 0, defect_detection_rate: 1 },
            ],
            sources: [
                { id: 'vs-src', name: 'Customer Orders', x: 30, y: 200, arrival_distribution: 'exponential', arrival_rate: 90, arrival_cv: 0.5, batch_size: 3, product_types: [{ name: 'Alpha', ratio: 0.5, color: '#4fc3f7', shelf_life: 0 }, { name: 'Beta', ratio: 0.5, color: '#ffb74d', shelf_life: 0 }], schedule_mode: 'batch_sequence', rush_order_rate: 0.05, due_date_target: 900, supplier_reliability: 1, incoming_quality_rate: 0, material_cost_per_unit: 0, late_delivery_penalty: 0 },
            ],
            sinks: [
                { id: 'vs-sink', name: 'Customer', x: 920, y: 200, sink_mode: 'counter' },
            ],
            connections: [
                { id: 'vs-c1', from_id: 'vs-src', to_id: 'vs-cut', buffer_capacity: 30, transport_type: 'hand_cart', transport_distance: 50 },
                { id: 'vs-c2', from_id: 'vs-cut', to_id: 'vs-drill', buffer_capacity: 30, transport_type: 'hand_cart', transport_distance: 40 },
                { id: 'vs-c3', from_id: 'vs-drill', to_id: 'vs-bend', buffer_capacity: 30, transport_type: 'hand_cart', transport_distance: 60 },
                { id: 'vs-c4', from_id: 'vs-bend', to_id: 'vs-assy', buffer_capacity: 30, transport_type: 'hand_cart', transport_distance: 45 },
                { id: 'vs-c5', from_id: 'vs-assy', to_id: 'vs-sink', buffer_capacity: null, transport_type: 'hand_cart', transport_distance: 30 },
            ],
            work_centers: [],
            operators: [],
            utility_systems: [],
        },
        config: { warmup: 300, runtime: 14400, speed: 100 },
        steps: [
            {
                title: 'Map the Current State',
                instruction: "Run the baseline. Total processing time (value-add) is only 65 seconds (15+12+18+20). But check the average lead time — it will be much higher. The ratio of value-add time to lead time is your process efficiency.",
                teaching: "In a value stream map, you draw the current state: every step, every queue, every delay. The shocking revelation is always the same: parts spend most of their time waiting, not being processed. A value-add ratio of 5-10% is common in batch-and-queue factories. The 7 wastes (muda): (1) Overproduction, (2) Waiting, (3) Transport, (4) Over-processing, (5) Inventory, (6) Motion, (7) Defects. This layout has oversized buffers (30 each), batch arrivals, transport delays between stations, and changeovers. Every one of these is visible waste.",
                highlightElements: ['m-leadtime', 'm-wip'],
            },
            {
                title: 'Eliminate Transport Waste',
                instruction: "Change all transport types to 'none' and distances to 0. This simulates moving the machines close together (cellular layout). Run again.",
                teaching: "Transport is pure waste — it adds zero value to the product. In a traditional functional layout, parts travel between departments: cutting department → drilling department → bending department. Each move involves forklifts, staging areas, tracking paperwork. A cell layout places machines in sequence, adjacent to each other. Parts flow directly from one operation to the next with no transport. Mike Rother's 'Learning to See' calls this creating continuous flow. The physical rearrangement eliminates an entire category of waste.",
                highlightElements: ['m-leadtime'],
            },
            {
                title: 'Reduce WIP (Inventory Waste)',
                instruction: "Reduce all buffer capacities from 30 to 5. This forces smaller batches and exposes problems. Run again — if stations block, you've found a process problem that WIP was hiding.",
                teaching: "Taiichi Ohno's river analogy: WIP is the water level. Problems (rocks) hide beneath the surface. Lower the water and the rocks become visible: machine imbalance, quality issues, unreliable processes. Large buffers are comfortable — they hide every problem. Small buffers are uncomfortable — they expose every problem. That's the point. A lean value stream has the minimum WIP needed to maintain flow, no more. The goal is not to optimize the buffer — it's to eliminate the need for it.",
                highlightElements: ['m-wip', 'm-leadtime'],
            },
            {
                title: 'Level the Flow (Eliminate Overproduction)',
                instruction: "Change source batch_size to 1 and schedule_mode to 'fixed_mix'. Reduce changeover times by half (SMED). Run again and compare lead time to the original baseline. How close is it to 65 seconds of pure processing?",
                teaching: "The future-state value stream map targets one-piece flow: a part enters the first station and exits the last station with minimal waiting. The ideal lead time equals the sum of processing times (65s). Everything above that is waste. By eliminating transport, reducing buffers, leveling the flow, and cutting changeovers, you've attacked 4 of the 7 wastes in one kaizen event. The remaining gap between actual lead time and 65s is variability (CV on each machine) — which standard work addresses. This is how value stream mapping drives transformation: see the waste, eliminate it systematically, measure the result.",
                highlightElements: ['m-leadtime', 'm-wip', 'm-throughput'],
            },
        ],
        challenges: [
            { description: 'Average lead time below 200 seconds', evaluate: (r) => r.avg_lead_time < 200 },
            { description: 'Average WIP below 8 parts', evaluate: (r) => r.avg_wip < 8 },
        ],
    },

    // =========================================================================
    // MISSIONS — Flight Simulator Grade Wargaming
    // =========================================================================
    {
        id: 'mission-cascade-failure',
        title: 'Cascade Failure',
        mode: 'mission',
        category: 'missions',
        difficulty: 3,
        estimatedMinutes: 12,
        overview: 'A routine Wednesday shift spirals as breakdowns cascade through a 4-machine line. Keep the customer order moving.',
        features: ['breakdowns', 'maintenance', 'dispatch_rules', 'operators'],
        briefing: 'Fort Worth Assembly Line 3 — Wednesday 06:00. The morning shift is starting. You have a 4-machine serial line producing actuator housings for an automotive OEM with a firm delivery window. Your maintenance crew is already stretched thin from overnight repairs on Line 1. At 08:15, the first machine will go down. What happens next depends on how fast you respond. Your operators are experienced but they follow YOUR calls. Every minute of indecision costs throughput. The customer does not care about your problems — they care about their 60 units.',
        config: { warmup: 300, runtime: 28800, speed: 100, maint_crew_size: 2 },
        layout: {
            stations: [
                { id: 'stn-1', name: 'CNC Rough', type: 'machine', x: 200, y: 200, process_time: 120, process_time_cv: 0.15, buffer_capacity: 20, queue: [],
                  mtbf: 5400, mttr: 900, mttr_cv: 0.3, needs_operator: true, priority: 1, dispatch_rule: 'FIFO' },
                { id: 'stn-2', name: 'CNC Finish', type: 'machine', x: 400, y: 200, process_time: 150, process_time_cv: 0.1, buffer_capacity: 15, queue: [],
                  mtbf: 7200, mttr: 600, mttr_cv: 0.2, needs_operator: true, priority: 2, dispatch_rule: 'FIFO' },
                { id: 'stn-3', name: 'Heat Treat', type: 'machine', x: 600, y: 200, process_time: 200, process_time_cv: 0.05, buffer_capacity: 10, queue: [],
                  mtbf: 14400, mttr: 1800, mttr_cv: 0.4, needs_operator: false, priority: 3, dispatch_rule: 'FIFO' },
                { id: 'stn-4', name: 'Final Inspect', type: 'machine', x: 800, y: 200, process_time: 90, process_time_cv: 0.2, buffer_capacity: 25, queue: [],
                  scrap_rate: 0.03, needs_operator: true, priority: 4, dispatch_rule: 'FIFO' },
            ],
            sources: [
                { id: 'src-1', name: 'Raw Material', x: 50, y: 200, inter_arrival: 130, inter_arrival_cv: 0.2 },
            ],
            sinks: [
                { id: 'sink-1', name: 'Shipping', x: 1000, y: 200 },
            ],
            connections: [
                { id: 'conn-1', from: 'src-1', to: 'stn-1' },
                { id: 'conn-2', from: 'stn-1', to: 'stn-2' },
                { id: 'conn-3', from: 'stn-2', to: 'stn-3' },
                { id: 'conn-4', from: 'stn-3', to: 'stn-4' },
                { id: 'conn-5', from: 'stn-4', to: 'sink-1' },
            ],
            operators: [
                { id: 'op-1', name: 'Ravi', assignedTo: 'stn-1', skill_level: 0.95 },
                { id: 'op-2', name: 'Priya', assignedTo: 'stn-2', skill_level: 0.9 },
                { id: 'op-3', name: 'Suresh', assignedTo: 'stn-4', skill_level: 0.85 },
            ],
            work_centers: [],
            utility_systems: [],
            shared_tools: [],
        },
        timeline: [
            // 08:15 — CNC Rough breaks down (scheduled MTBF but forced here for drama)
            { at: 1800, type: 'mission_event', target: 'stn-1', severity: 'critical',
              message: 'CNC Rough — spindle bearing failure. Machine down.',
              effect: { type: 'breakdown' } },
            // 09:00 — Quality excursion on CNC Finish (scrap spike)
            { at: 4500, type: 'mission_event', target: 'stn-2', severity: 'warning',
              message: 'CNC Finish — dimensional drift detected. Scrap rate rising.',
              effect: { type: 'quality_excursion', scrap_rate: 0.15, duration: 2400 } },
            // 10:30 — Demand spike (customer expedites)
            { at: 9900, type: 'mission_event', target: 'src-1', severity: 'warning',
              message: 'Customer expedite: arrival rate doubled for next 2 hours.',
              effect: { type: 'demand_spike', multiplier: 2.0, duration: 7200 } },
            // 12:00 — Operator quits (lunch and doesn't come back)
            { at: 14400, type: 'mission_event', target: null, severity: 'critical',
              message: 'Operator Suresh has left the building. Final Inspect unstaffed.',
              effect: { type: 'operator_quit', operator_id: 'op-3' } },
            // 14:00 — Heat Treat breaks down
            { at: 21600, type: 'mission_event', target: 'stn-3', severity: 'critical',
              message: 'Heat Treat — thermocouple failure. Furnace cycling off.',
              effect: { type: 'breakdown' } },
        ],
        scoring: {
            objectives: [
                'Deliver 60+ units by end of shift',
                'Maintain yield above 90%',
                'Respond to all critical events within 120 seconds',
                'Keep total cost under $3000',
            ],
            weights: {
                throughput: { target: 60, points: 30 },
                yield: { target: 0.90, points: 25 },
                response_time: { target: 120, points: 25 },
                cost: { target: 3000, points: 20 },
            },
        },
        debrief: [
            { topic: 'Spindle Bearing Failure (08:15)',
              analysis: 'The CNC Rough going down starves the entire line. Every second it stays down, downstream machines sit idle. This is your constraint during the crisis.',
              optimalAction: 'Immediately request maintenance priority for CNC Rough. If crew is busy, force-repair at premium cost. Reassign Ravi to CNC Finish to support Priya during the downtime.' },
            { topic: 'Quality Excursion (09:00)',
              analysis: 'A 15% scrap rate on CNC Finish means 1 in 7 parts is waste. With the line already starved, every scrapped part hurts double.',
              optimalAction: 'Change dispatch rule to SPT on CNC Finish to clear good parts faster. If scrap is unacceptable, quarantine and stop the machine for a tool change — but only if CNC Rough is still down (no parts coming anyway).' },
            { topic: 'Customer Expedite (10:30)',
              analysis: 'Double arrival rate with a recovering line creates a WIP explosion at the bottleneck. The buffer fills, upstream blocks, and lead time spikes.',
              optimalAction: 'Authorize overtime to extend capacity. Switch Final Inspect dispatch to EDD (Earliest Due Date) to prioritize expedited work.' },
            { topic: 'Operator Walkout (12:00)',
              analysis: 'Losing Suresh at Final Inspect creates an immediate bottleneck at the end of the line. WIP backs up through Heat Treat.',
              optimalAction: 'Reassign Ravi or Priya to Final Inspect. The upstream machine they leave will starve, but the constraint has shifted to inspection. One operator covering two stations with toggling is better than a dead station.' },
            { topic: 'Heat Treat Failure (14:00)',
              analysis: 'Late-shift equipment failure with depleted maintenance capacity. The line is now broken at two points.',
              optimalAction: 'Force-repair Heat Treat at premium cost. With only 4 hours left in the shift, every minute of downtime directly reduces output. This is where you spend money to make the delivery window.' },
        ],
        steps: [],
        challenges: [],
    },
];

function openScenarioLauncher() {
    renderScenarioLauncher();
    document.getElementById('scenario-launcher').classList.add('active');
}
function closeScenarioLauncher() {
    document.getElementById('scenario-launcher').classList.remove('active');
}

function renderScenarioLauncher() {
    const container = document.getElementById('scenario-tiers');
    if (typeof GUIDED_SCENARIOS === 'undefined' || !GUIDED_SCENARIOS.length) {
        container.innerHTML = '<p style="color:var(--text-dim); font-size:0.75rem;">No scenarios available.</p>';
        return;
    }
    // Group by category
    const categories = {};
    const categoryLabels = {
        foundations: 'Foundations',
        variability: 'Variability & Reliability',
        workforce: 'Workforce',
        quality: 'Quality',
        systems: 'Systems Complexity',
        mastery: 'Mastery',
        missions: 'Missions',
    };
    for (const s of GUIDED_SCENARIOS) {
        const cat = s.category || 'other';
        if (!categories[cat]) categories[cat] = [];
        categories[cat].push(s);
    }
    let html = '';
    for (const [cat, scenarios] of Object.entries(categories)) {
        const label = categoryLabels[cat] || cat;
        html += `<div class="scenario-tier">`;
        html += `<div class="scenario-tier-label">${label}</div>`;
        html += `<div class="scenario-grid">`;
        for (const s of scenarios) {
            const progress = scenarioProgress[s.id];
            const done = progress?.completed;
            const isMission = s.mode === 'mission';
            const clickFn = isMission ? `loadMission('${s.id}')` : `loadScenario('${s.id}')`;
            const dots = Array.from({length: 5}, (_, i) =>
                `<span class="scenario-dot ${i < s.difficulty ? 'filled' : ''}"></span>`
            ).join('');
            const tags = (s.features || []).slice(0, 4).map(f =>
                `<span class="scenario-tag">${f}</span>`
            ).join('');
            const badge = isMission ? '<span style="background:#e74c3c;color:#fff;font-size:0.55rem;padding:1px 5px;border-radius:3px;margin-left:6px;vertical-align:middle;font-weight:700;letter-spacing:0.5px;">MISSION</span>' : '';
            const gradeDisplay = isMission && done && progress.grade ? ` <span style="font-weight:700;color:${progress.grade === 'A' ? '#4a9f6e' : progress.grade === 'B' ? '#3b82f6' : '#f59e0b'};">${progress.grade}</span>` : '';
            html += `<div class="scenario-card ${done ? 'completed' : ''}" onclick="${clickFn}">
                <div class="scenario-card-title">${s.title}${badge}${done ? ' <span class="check">&#10003;</span>' : ''}${gradeDisplay}</div>
                <div class="scenario-card-desc">${s.overview}</div>
                <div class="scenario-card-meta">
                    <span class="scenario-difficulty">${dots}</span>
                    <span>~${s.estimatedMinutes} min</span>
                </div>
                ${tags ? `<div class="scenario-features" style="margin-top:4px;">${tags}</div>` : ''}
            </div>`;
        }
        html += `</div></div>`;
    }
    container.innerHTML = html;
}

function loadScenario(scenarioId) {
    const scenario = (typeof GUIDED_SCENARIOS !== 'undefined' ? GUIDED_SCENARIOS : []).find(s => s.id === scenarioId);
    if (!scenario) return;

    // Stop any running sim
    if (des && des.running) {
        des.running = false;
        if (animFrameId) cancelAnimationFrame(animFrameId);
    }
    des = null;
    currentSimId = null;

    // Deep-clone and load layout
    const sl = JSON.parse(JSON.stringify(scenario.layout));
    layout.stations = sl.stations || [];
    layout.sources = sl.sources || [];
    layout.sinks = sl.sinks || [];
    layout.connections = sl.connections || [];
    layout.work_centers = sl.work_centers || [];
    layout.operators = sl.operators || [];
    layout.utility_systems = sl.utility_systems || [];
    layout.shared_tools = sl.shared_tools || [];

    document.getElementById('sim-name').value = `Scenario: ${scenario.title}`;

    // Apply config
    if (scenario.config) {
        if (scenario.config.warmup != null) document.getElementById('cfg-warmup').value = scenario.config.warmup;
        if (scenario.config.runtime != null) document.getElementById('cfg-runtime').value = scenario.config.runtime;
        if (scenario.config.speed != null) {
            document.getElementById('cfg-speed').value = scenario.config.speed;
            document.getElementById('cfg-speed-label').textContent = scenario.config.speed + 'x';
        }
        if (scenario.config.maint_crew_size != null) document.getElementById('cfg-maint-crew').value = scenario.config.maint_crew_size;
        if (scenario.config.agv_fleet_size != null) document.getElementById('cfg-agv-fleet').value = scenario.config.agv_fleet_size;
    }

    // Reset nextId past scenario IDs
    const allEls = [...layout.stations, ...layout.sources, ...layout.sinks, ...layout.connections];
    for (const el of allEls) {
        const num = parseInt(String(el.id).split('-').pop());
        if (!isNaN(num) && num >= nextId) nextId = num + 1;
    }

    // Activate
    activeScenario = scenario;
    activeStepIndex = 0;
    scenarioChallengeResults = null;
    savedRuns = [];

    renderCanvas();
    if (typeof renderOperatorList === 'function') renderOperatorList();
    if (typeof renderUtilityList === 'function') renderUtilityList();
    if (typeof renderToolList === 'function') renderToolList();
    resetView();

    showTeachingPanel();
    renderCurrentStep();
    applyStepHighlights();
    closeScenarioLauncher();
    showToast(`Scenario loaded: ${scenario.title}`);
}

function showTeachingPanel() {
    document.getElementById('teaching-panel').classList.add('active');
}

function hideTeachingPanel() {
    document.getElementById('teaching-panel').classList.remove('active');
    document.getElementById('teaching-restore-btn').style.display = 'none';
}
function collapseTeachingPanel(collapse) {
    document.getElementById('teaching-panel').classList.toggle('active', !collapse);
    document.getElementById('teaching-restore-btn').style.display = collapse ? '' : 'none';
}

function renderCurrentStep() {
    if (!activeScenario) return;
    const step = activeScenario.steps[activeStepIndex];
    if (!step) return;

    document.getElementById('teaching-title').textContent = activeScenario.title;

    // Progress dots
    document.getElementById('teaching-progress').innerHTML = activeScenario.steps.map((s, i) =>
        `<span class="step-dot ${i === activeStepIndex ? 'active' : i < activeStepIndex ? 'done' : ''}"
               onclick="goToScenarioStep(${i})" title="${s.title}"></span>`
    ).join('');

    // Step content
    document.getElementById('teaching-step-content').innerHTML = `
        <div class="teaching-step-title">Step ${activeStepIndex + 1}: ${step.title}</div>
        <div class="teaching-instruction">${step.instruction}</div>
        <div class="teaching-why-label">Why this matters</div>
        <div class="teaching-why-text">${step.teaching}</div>
    `;

    // Challenges
    renderScenarioChallenges();

    // Nav
    document.getElementById('teaching-prev').disabled = activeStepIndex === 0;
    const nextBtn = document.getElementById('teaching-next');
    if (activeStepIndex === activeScenario.steps.length - 1) {
        nextBtn.textContent = 'Complete';
    } else {
        nextBtn.innerHTML = 'Next &#9654;';
    }
}

function renderScenarioChallenges() {
    const el = document.getElementById('teaching-challenges');
    if (!activeScenario || !activeScenario.challenges || activeScenario.challenges.length === 0) {
        el.innerHTML = '';
        return;
    }
    el.innerHTML = `
        <div class="teaching-challenges-label">Challenges</div>
        ${activeScenario.challenges.map((c, i) => {
            const r = scenarioChallengeResults ? scenarioChallengeResults[i] : null;
            const icon = r ? (r.passed ? '&#10003;' : '&#10007;') : '&#9675;';
            const cls = r ? (r.passed ? 'passed' : 'failed') : '';
            return `<div class="challenge-row ${cls}">
                <span class="challenge-icon">${icon}</span>
                <span>${c.description}</span>
            </div>`;
        }).join('')}
    `;
}

function applyStepHighlights() {
    // Clear previous
    document.querySelectorAll('.scenario-highlight').forEach(el => el.classList.remove('scenario-highlight'));
    document.querySelectorAll('.scenario-dimmed').forEach(el => el.classList.remove('scenario-dimmed'));

    if (!activeScenario) return;
    const step = activeScenario.steps[activeStepIndex];
    if (!step) return;

    // Highlight specific elements
    if (step.highlightElements) {
        for (const elId of step.highlightElements) {
            const el = document.getElementById(elId);
            if (el) el.classList.add('scenario-highlight');
        }
    }

    // Dim irrelevant sidebar sections
    if (step.highlightSections && step.highlightSections.length > 0) {
        document.querySelectorAll('.sidebar-section').forEach(section => {
            const h3 = section.querySelector('h3');
            if (h3 && !step.highlightSections.includes(h3.textContent.trim())) {
                section.classList.add('scenario-dimmed');
            }
        });
    }
}

function scenarioNextStep() {
    if (!activeScenario) return;
    if (activeStepIndex < activeScenario.steps.length - 1) {
        activeStepIndex++;
        renderCurrentStep();
        applyStepHighlights();
    } else {
        // On final step — mark complete
        scenarioProgress[activeScenario.id] = { completed: true, completedAt: Date.now() };
        saveScenarioProgress();
        showToast('Scenario complete!');
    }
}

function scenarioPrevStep() {
    if (!activeScenario || activeStepIndex <= 0) return;
    activeStepIndex--;
    renderCurrentStep();
    applyStepHighlights();
}

function goToScenarioStep(i) {
    if (!activeScenario || i < 0 || i >= activeScenario.steps.length) return;
    activeStepIndex = i;
    renderCurrentStep();
    applyStepHighlights();
}

function evaluateScenarioChallenges(results) {
    if (!activeScenario || !activeScenario.challenges) return;
    scenarioChallengeResults = activeScenario.challenges.map(c => ({
        description: c.description,
        passed: c.evaluate(results),
    }));
    renderScenarioChallenges();

    const allPassed = scenarioChallengeResults.every(c => c.passed);
    if (allPassed) {
        scenarioProgress[activeScenario.id] = { completed: true, completedAt: Date.now() };
        saveScenarioProgress();
        showToast('All challenges passed!');
    }
}

function exitScenario() {
    activeScenario = null;
    activeStepIndex = 0;
    scenarioChallengeResults = null;
    hideTeachingPanel();

    // Clear highlights
    document.querySelectorAll('.scenario-highlight').forEach(el => el.classList.remove('scenario-highlight'));
    document.querySelectorAll('.scenario-dimmed').forEach(el => el.classList.remove('scenario-dimmed'));

    showToast('Scenario exited');
}

