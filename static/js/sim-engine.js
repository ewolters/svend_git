// =============================================================================
// DES Engine
// =============================================================================

class PlantDES {
    constructor(layout) {
        this.eventQueue = new MinHeap();
        this.clock = 0;
        this.stations = new Map();
        this.adjacency = new Map();   // fromId -> [{toId, bufferCapacity}]
        this.reverseAdj = new Map();  // toId -> [fromId]
        this.jobs = [];
        this.completedJobs = [];
        this.jobIdCounter = 0;
        this.warmupTime = parseFloat(document.getElementById('cfg-warmup').value) || 300;
        this.runTime = parseFloat(document.getElementById('cfg-runtime').value) || 3600;
        this.running = false;
        this.history = { time: [], wip: [], throughput: [] };
        this.stationHistory = {};

        // Operator pool
        this.operators = [];
        this.calloffRate = parseFloat(document.getElementById('cfg-calloff').value) / 100 || 0.05;
        this.quitRate = parseFloat(document.getElementById('cfg-quit').value) / 100 || 0.005;
        this.workforceLog = []; // [{time, event, operator, detail}]

        // Cost accounting
        this.costConfig = {
            laborCostPerHour: parseFloat(document.getElementById('cfg-labor-cost')?.value) || 25,
            overtimePremium: parseFloat(document.getElementById('cfg-ot-premium')?.value) || 1.5,
            holdingCostPerUnitHour: parseFloat(document.getElementById('cfg-holding-cost')?.value) || 0.5,
        };
        this.costs = {
            labor: 0,           // total labor $ (regular + overtime)
            material: 0,        // total material $ from sources
            scrapWaste: 0,      // material $ lost to scrap
            holdingCost: 0,     // WIP holding cost
            overtimeCost: 0,    // overtime premium portion
            totalCost: 0,       // sum of all
        };

        // Escaped defect tracking
        this.escapedDefects = 0;        // defects that made it to customer
        this.detectedDownstream = 0;    // defects caught at a later station
        this.customerReturns = 0;       // defects that reached the sink undetected

        // Rush order / due date tracking
        this.rushOrderCount = 0;
        this.lateOrderCount = 0;
        this.onTimeCount = 0;
        this.promotedToRush = 0;  // orders that became rush mid-process (the spiral)
        this.expiredWIP = 0;           // jobs scrapped due to shelf life expiry

        // Overtime authorization
        this.overtimeConfig = {
            wipThreshold: parseInt(document.getElementById('cfg-ot-wip-threshold')?.value) || 0,  // 0 = disabled
            maxOTHours: parseFloat(document.getElementById('cfg-ot-max-hours')?.value) || 4,
        };
        this.overtimeActive = false;
        this.overtimeShifts = 0;     // how many shifts have had OT authorized
        this.totalOTHours = 0;       // total overtime hours across all operators

        // Customer behavior: satisfaction drives order flow and revenue
        this.customerConfig = {
            revenuePerUnit: parseFloat(document.getElementById('cfg-revenue-per-unit')?.value) || 0,  // 0 = disabled
            returnCostMultiplier: 3,  // returns cost 3× revenue (warranty, shipping, goodwill)
        };
        this.customerSatisfaction = 1.0;  // 0.0–1.0 — starts perfect
        this.totalRevenue = 0;
        this.totalReturnCost = 0;
        this.customerLostOrders = 0;     // orders that didn't happen because satisfaction is low

        // Management reactivity: auto-reactive policy oscillation
        const mgmtReactivity = document.getElementById('cfg-mgmt-reactivity')?.value || 'off';
        this.management = {
            reactivity: mgmtReactivity,  // off, calm, nervous, panicking
            reviewInterval: mgmtReactivity === 'panicking' ? 1800 : mgmtReactivity === 'nervous' ? 3600 : 7200,
            lastWIP: 0,
            lastYield: 1,
            lastOnTime: 1,
            policyLog: [],  // [{time, action, detail}]
            interventionCount: 0,
        };
        if (mgmtReactivity !== 'off') {
            this.eventQueue.push({ time: this.management.reviewInterval, type: 'management_review' });
        }

        // Shared utility systems — correlated failures
        this.utilitySystems = [];
        const utilityConfigs = layout.utility_systems || [];
        for (const u of utilityConfigs) {
            this.utilitySystems.push({
                id: u.id,
                name: u.name || 'Utility',
                mtbf: u.mtbf || 0,          // 0 = never fails
                mttr: u.mttr || 300,         // default 5min repair
                affectedMachines: u.affected_machines || [],  // station IDs
                state: 'up',                 // up | down
                failureCount: 0,
                totalDowntime: 0,
                downSince: null,
            });
        }

        // AGV / material handling fleet — finite transport resource
        const agvFleetSize = parseInt(document.getElementById('cfg-agv-fleet')?.value) || 0;
        this.agvFleet = {
            size: agvFleetSize,         // 0 = unlimited (instant transport scheduling)
            available: agvFleetSize,    // currently free AGVs
            transportQueue: [],         // [{job, fromId, toId, transportTime, requestedAt}]
            totalWaitTime: 0,
            tripsCompleted: 0,
        };

        // Inspector pool — finite quality inspection resource
        const inspPoolSize = parseInt(document.getElementById('cfg-inspector-pool')?.value) || 0;
        this.inspectorPool = {
            size: inspPoolSize,         // 0 = unlimited (inline quality check)
            available: inspPoolSize,
            inspectionQueue: [],        // [{job, stationId, requestedAt}]
            totalWaitTime: 0,
            inspectionsCompleted: 0,
            escapedDueToRush: 0,        // defects that escaped because inspection was skipped under pressure
        };

        // Engineering Change Orders — random mid-production spec changes
        const ecoRate = parseFloat(document.getElementById('cfg-eco-rate')?.value) || 0;
        this.ecoConfig = {
            rate: ecoRate,  // ECOs per simulated hour, 0 = disabled
        };
        this.ecoCount = 0;
        this.ecoScrappedWIP = 0;
        if (ecoRate > 0) {
            const firstEco = this._sampleExponential(3600 / ecoRate);
            this.eventQueue.push({ time: firstEco, type: 'eco_event' });
        }

        // Shared tooling / fixtures — finite shared tools
        this.sharedTools = [];
        const toolConfigs = layout.shared_tools || [];
        for (const t of toolConfigs) {
            this.sharedTools.push({
                id: t.id,
                name: t.name || 'Fixture',
                copies: t.copies || 1,
                available: t.copies || 1,
                requiredBy: t.required_by || [],  // station IDs
                toolQueue: [],   // [{stationId, requestedAt}]
                totalWaitTime: 0,
                useCount: 0,
            });
        }

        // Shared maintenance crew — finite repair resource
        const crewSize = parseInt(document.getElementById('cfg-maint-crew')?.value) || 0;
        this.maintenanceCrew = {
            size: crewSize,             // 0 = unlimited (instant repair), >0 = finite crew
            available: crewSize,        // currently free technicians
            repairQueue: [],            // machines waiting for a technician [{stationId, requestedAt, repairTime}]
            totalWaitTime: 0,           // cumulative time machines spent waiting for crew
            repairsCompleted: 0,
        };

        // ===== MISSION MODE ENGINE PROPERTIES =====
        this.missionMode = false;
        this.missionTimeline = [];
        this.missionAlerts = [];
        this.decisionLog = [];
        this.alertIdCounter = 0;
        this.paused = false;
        this.onAlert = null;
        this.onPause = null;

        this._buildFromLayout(layout);
    }

    _buildFromLayout(layout) {
        // Sources
        for (const src of layout.sources || []) {
            this.stations.set(src.id, {
                id: src.id, type: 'source', name: src.name || 'Source',
                x: src.x, y: src.y,
                arrivalRate: src.arrival_rate || 60,
                arrivalDist: src.arrival_distribution || 'exponential',
                arrivalCV: src.arrival_cv || 0,
                batchSize: src.batch_size || 1,
                productTypes: src.product_types || [{ name: 'A', ratio: 1.0, color: '#4a9f6e' }],
                scheduleMode: src.schedule_mode || 'fixed_mix',
                currentBatchType: null,  // For batch_sequence mode
                currentBatchRemaining: 0,
                // Rush order config
                rushOrderRate: src.rush_order_rate || 0,
                dueDateTarget: src.due_date_target || 0,
                latenessThreshold: src.lateness_threshold || 0.8,
                // Supplier variability
                supplierReliability: src.supplier_reliability ?? 1.0,  // 1.0 = always on time, 0.7 = 30% chance of late delivery
                lateDeliveryPenalty: src.late_delivery_penalty || 0,   // extra seconds added when supplier is late
                incomingQualityRate: src.incoming_quality_rate ?? 1.0, // 1.0 = perfect material, 0.9 = 10% chance of bad batch
                materialCostPerUnit: src.material_cost_per_unit || 0, // $ per unit at source
                state: 'active', queue: [], stats: this._emptyStats(),
            });
        }

        // Machines
        for (const stn of layout.stations || []) {
            this.stations.set(stn.id, {
                id: stn.id, type: 'machine', name: stn.name || 'Machine',
                x: stn.x, y: stn.y,
                cycleTime: stn.cycle_time || 30,
                cycleTimeCV: stn.cycle_time_cv || 0,
                changeoverTime: stn.changeover_time || 0,
                changeoverFreq: stn.changeover_frequency || 0,
                uptime: stn.uptime || 100,
                mtbf: stn.mtbf || null,
                mttr: stn.mttr || null,
                operators: stn.operators || 1,
                batchSize: stn.batch_size || 1,
                dedicatedProduct: stn.dedicated_product || null,
                workCenterId: stn.work_center_id || null,
                scrapRate: stn.scrap_rate || 0,
                defectRate: stn.defect_rate || 0,
                reworkRate: stn.rework_rate || 0,
                qualityByProduct: stn.quality_by_product || {},
                shiftSchedule: stn.shift_schedule || '24/7',
                breakDuration: stn.break_duration || 0,
                setupMatrix: stn.setup_matrix || {},
                // Weibull degradation config
                weibullBeta: stn.weibull_beta || 1,
                pmInterval: stn.pm_interval || 0,
                pmDuration: stn.pm_duration || 0,
                // Escaped defects: probability that a defect is DETECTED at this station
                defectDetectionRate: stn.defect_detection_rate ?? 1.0,  // 1.0 = perfect inspection, 0.8 = 20% escape
                // Process drift: scrap rate creeps up over time between calibrations
                driftRate: stn.drift_rate || 0,            // scrap rate increase per 100 jobs processed (e.g., 0.01 = +1% per 100 jobs)
                calibrationInterval: stn.calibration_interval || 0,  // auto-calibrate every N jobs, 0=manual only
                driftAccumulated: 0,                        // current drift amount added to scrap
                jobsSinceCalibration: 0,                   // jobs since last calibration
                operatingTime: 0,                      // cumulative processing time since last maintenance
                pmCount: 0,                            // PM events completed
                unplannedDownCount: 0,                 // breakdown count
                // Shift handover loss
                handoverLossRate: stn.handover_loss_rate || 0,   // probability of repeated setup at shift change (0-1)
                firstArticlePenalty: stn.first_article_penalty || 0,  // extra scrap % on first N parts after idle
                firstArticleCount: stn.first_article_count || 3,     // how many parts are "first article" after warmup
                warmupTime: stn.warmup_time || 0,                    // seconds of warmup needed after idle
                idleSince: 0,                                         // when machine last went idle
                partsAfterRestart: 999,                               // parts since last restart (high = past first-article window)
                handoverLossCount: 0,                                  // count of handover-induced setups
                // Cost tracking
                laborCostPerHour: stn.labor_cost_per_hour || 0,
                materialCostPerUnit: stn.material_cost_per_unit || 0,
                // Dispatch rule: FIFO, SPT, EDD, CR, WSPT
                dispatchRule: stn.dispatch_rule || 'FIFO',
                _utilityDown: false,  // flag: machine downed by utility system failure
                _holdingTool: null,  // ID of shared tool currently held
                // Batch processing (oven/furnace): accumulate N parts, process all at once
                batchProcessMode: stn.batch_process_mode || false,
                batchProcessSize: stn.batch_process_size || 10,    // wait for N parts
                batchProcessTime: stn.batch_process_time || 300,   // fixed time to process the whole batch
                // Assembly mode: requires multiple input part types before processing
                assemblyMode: stn.assembly_mode || false,
                assemblyInputs: stn.assembly_inputs || {},  // { productType: qty_needed }
                assemblyBuffer: {},  // { productType: [jobs] } — collected parts waiting for assembly
                // Micro-stoppages: brief stops that hide in processing time
                microStopRate: stn.micro_stop_rate || 0,      // probability per cycle (0-1)
                microStopDuration: stn.micro_stop_duration || 10, // avg seconds per micro-stop
                microStopCount: 0,                              // count for tracking
                microStopLostTime: 0,                           // total hidden lost time
                // Contamination: residue from previous product affects next
                contaminationRisk: stn.contamination_risk || 0,      // 0-1 probability after quick changeover
                thoroughCleanMultiplier: stn.thorough_clean_multiplier || 2, // how much longer a thorough clean takes
                _contaminated: false,                                 // current contamination state
                contaminationEvents: 0,                               // tracking
                // Rework routing: send rework to a different station instead of self
                reworkTarget: stn.rework_target || null,  // station ID or null (self)
                // Inline SPC: control chart monitoring
                spcEnabled: stn.spc_enabled || false,
                spcInvestigationTime: stn.spc_investigation_time || 120,  // seconds to investigate signal
                spcMeasurements: [],      // rolling window of quality measurements
                spcSignalCount: 0,        // total out-of-control signals
                spcFalseAlarms: 0,        // signals that weren't real (frustrating!)
                // Measurement lag: quality data arrives late
                measurementDelay: stn.measurement_delay || 0,  // seconds before measurement hits SPC chart
                _pendingMeasurements: [],  // [{value, availableAt}] — delayed measurements
                containmentScope: 0,       // parts produced between defect start and detection
                state: 'idle', queue: [], currentJob: null,
                lastProductType: null, blockedJob: null,
                stats: this._emptyStats(),
                stateStartTime: 0,
            });
        }

        // Sinks
        for (const sink of layout.sinks || []) {
            const sinkMode = sink.sink_mode || 'exit';
            const demandRates = (sink.demand_rates || []).map(d => ({
                product: d.product,
                rate: d.rate || 10,  // per hour
                distribution: d.distribution || 'exponential',
                cv: d.cv || 0,
            }));
            const safetyStock = sink.safety_stock || {};
            const reorderPoint = sink.reorder_point || {};
            this.stations.set(sink.id, {
                id: sink.id, type: 'sink', name: sink.name || 'Sink',
                x: sink.x, y: sink.y,
                sinkMode,
                fgInventory: {},     // { productType: count }
                demandRates,
                safetyStock,         // { productType: minLevel }
                reorderPoint,        // { productType: triggerLevel }
                stockouts: {},       // { productType: count }
                demandFilled: {},    // { productType: count }
                fgHistory: [],       // [{time, inventory: {...}}]
                state: 'active', queue: [], stats: this._emptyStats(),
            });
        }

        // Connections → adjacency map
        for (const conn of layout.connections || []) {
            if (!this.adjacency.has(conn.from_id)) this.adjacency.set(conn.from_id, []);
            this.adjacency.get(conn.from_id).push({
                toId: conn.to_id,
                bufferCapacity: conn.buffer_capacity,
                transportType: conn.transport_type || 'none',
                transportDistance: conn.transport_distance || 0,
                routingRules: conn.routing_rules || [],
            });
            if (!this.reverseAdj.has(conn.to_id)) this.reverseAdj.set(conn.to_id, []);
            this.reverseAdj.get(conn.to_id).push(conn.from_id);
        }

        // Init: schedule first arrivals and breakdowns
        for (const [id, stn] of this.stations) {
            if (stn.type === 'source') {
                const t = this._sampleInterarrival(stn);
                this.eventQueue.push({ time: t, type: 'arrival', stationId: id });
            }
            if (stn.type === 'machine' && stn.mtbf && stn.mtbf > 0) {
                const t = this._sampleWeibull(stn.mtbf, stn.weibullBeta || 1);
                this.eventQueue.push({ time: t, type: 'breakdown', stationId: id });
            }
            // Schedule first PM if configured
            if (stn.type === 'machine' && stn.pmInterval > 0) {
                this.eventQueue.push({ time: stn.pmInterval, type: 'pm_start', stationId: id });
            }
        }

        // Schedule utility system failures
        for (const util of this.utilitySystems) {
            if (util.mtbf > 0) {
                const t = this._sampleExponential(util.mtbf);
                this.eventQueue.push({ time: t, type: 'utility_failure', utilityId: util.id });
            }
        }

        // Schedule shift breaks for machines with schedules
        for (const [id, stn] of this.stations) {
            if (stn.type === 'machine' && stn.shiftSchedule !== '24/7') {
                this._scheduleShiftEvents(stn);
            }
        }

        // Build operator pool
        for (const op of layout.operators || []) {
            const desOp = {
                id: op.id,
                name: op.name,
                skills: { ...(op.skills || {}) },
                status: op.status === 'quit' ? 'quit' : 'available',
                assignedTo: null,
                trainingTarget: op._trainingTarget || null,
                trainingEndsAt: op._trainingTarget ? (op._trainingHours || 4) * 3600 : null,
                // Fatigue & morale
                fatigue: 0,          // 0-1, increases over shift, affects speed + error rate
                morale: 0.8,         // 0-1, affected by peer quits, overtime, breakdowns
                shiftStart: 0,       // when current shift started
                overtimeHours: 0,    // accumulated overtime this sim
                jobsThisShift: 0,    // jobs completed this shift
                // Skill forgetting: track when each station was last worked
                lastWorkedAt: {},    // { stationId: simClock } — for decay calculation
                // Tribal knowledge: undocumented process tweaks learned through experience
                processTweaks: {},   // { stationId: 0.0-1.0 } — accumulated tricks that improve yield + speed
            };

            // Handle training: operator unavailable until training completes
            if (desOp.trainingTarget) {
                desOp.status = 'training';
                this.eventQueue.push({
                    time: desOp.trainingEndsAt,
                    type: 'training_complete',
                    operatorId: desOp.id,
                    stationId: desOp.trainingTarget,
                });
            }

            this.operators.push(desOp);
        }

        // Schedule workforce events (call-offs, quits) at shift boundaries
        if (this.operators.length > 0) {
            // Check attendance every 8 hours (shift change)
            const endTime = this.warmupTime + this.runTime;
            for (let t = 0; t < endTime; t += 28800) {
                this.eventQueue.push({ time: t, type: 'workforce_check' });
            }
        }

        // Schedule demand events for FG warehouse sinks
        for (const [id, stn] of this.stations) {
            if (stn.type === 'sink' && stn.sinkMode === 'fg_warehouse' && stn.demandRates.length > 0) {
                for (const dr of stn.demandRates) {
                    if (dr.rate > 0) {
                        const interarrival = 3600 / dr.rate; // rate is per hour
                        const firstDemand = this._sampleExponential(interarrival);
                        this.eventQueue.push({
                            time: firstDemand,
                            type: 'customer_demand',
                            stationId: id,
                            product: dr.product,
                        });
                    }
                }
            }
        }

        // Schedule periodic schedule review for kanban/chase sources (every 5 min sim time)
        const hasDynamicSource = [...this.stations.values()].some(s =>
            s.type === 'source' && (s.scheduleMode === 'kanban' || s.scheduleMode === 'chase_demand')
        );
        if (hasDynamicSource) {
            this.eventQueue.push({ time: 60, type: 'schedule_review' });
        }

        // Schedule rush order expediting check (scans WIP for late orders every 60s)
        const hasDueDates = [...this.stations.values()].some(s =>
            s.type === 'source' && s.dueDateTarget > 0
        );
        if (hasDueDates) {
            this.eventQueue.push({ time: 60, type: 'rush_check' });
        }

        // Schedule periodic stats recording
        this.eventQueue.push({ time: 10, type: 'record_stats' });

        // Schedule mission timeline events
        if (this.missionTimeline.length > 0) {
            for (const te of this.missionTimeline) {
                te.fired = false;
                this.eventQueue.push({
                    time: this.warmupTime + te.at,
                    type: 'mission_event',
                    timelineEntry: te,
                });
            }
        }
    }

    // Shift schedule definitions: returns array of {start, end} work windows per cycle (seconds)
    // All times relative to a repeating cycle
    _getShiftWindows(schedule) {
        switch (schedule) {
            case 'single-8':
                // 8hr work, 30min lunch at 4hr mark, 16hr off
                // Cycle: 86400s (24hr)
                return { cycle: 86400, windows: [
                    { start: 0, end: 14400 },        // 0-4hr work
                    { start: 16200, end: 28800 },     // 4.5hr-8hr work (30min lunch)
                ] };
            case 'double-16':
                // Two 8hr shifts, 30min break each, 8hr off
                return { cycle: 86400, windows: [
                    { start: 0, end: 14400 },
                    { start: 16200, end: 28800 },     // Shift 1: 0-8hr w/ lunch
                    { start: 28800, end: 43200 },
                    { start: 45000, end: 57600 },     // Shift 2: 8-16hr w/ lunch
                ] };
            case 'triple-24':
                // Three 8hr shifts, 30min break each
                return { cycle: 86400, windows: [
                    { start: 0, end: 14400 },
                    { start: 16200, end: 28800 },
                    { start: 28800, end: 43200 },
                    { start: 45000, end: 57600 },
                    { start: 57600, end: 72000 },
                    { start: 73800, end: 86400 },
                ] };
            case 'single-12':
                // 12hr shift, 45min lunch at 6hr
                return { cycle: 86400, windows: [
                    { start: 0, end: 21600 },
                    { start: 24300, end: 43200 },
                ] };
            default:
                return null; // 24/7, no breaks
        }
    }

    _scheduleShiftEvents(stn) {
        const schedule = this._getShiftWindows(stn.shiftSchedule);
        if (!schedule) return;

        const { cycle, windows } = schedule;
        const endTime = this.warmupTime + this.runTime;

        // Schedule all break_start / break_end pairs across the sim duration
        for (let offset = 0; offset < endTime + cycle; offset += cycle) {
            let prevEnd = 0;
            for (const win of windows) {
                // Gap before this window = break
                if (win.start > prevEnd) {
                    const breakStart = offset + prevEnd;
                    const breakEnd = offset + win.start;
                    if (breakStart < endTime) {
                        this.eventQueue.push({ time: breakStart, type: 'break_start', stationId: stn.id });
                        this.eventQueue.push({ time: breakEnd, type: 'break_end', stationId: stn.id });
                    }
                }
                prevEnd = win.end;
            }
            // Gap after last window to end of cycle = break
            if (prevEnd < cycle) {
                const breakStart = offset + prevEnd;
                const breakEnd = offset + cycle;
                if (breakStart < endTime) {
                    this.eventQueue.push({ time: breakStart, type: 'break_start', stationId: stn.id });
                    this.eventQueue.push({ time: breakEnd, type: 'break_end', stationId: stn.id });
                }
            }
        }
    }

    _emptyStats() {
        return { processing: 0, setup: 0, down: 0, plannedDown: 0, starved: 0, blocked: 0, idle: 0, onBreak: 0, processed: 0, scrapped: 0, reworked: 0, changeovers: 0, queueSum: 0, queueSamples: 0 };
    }

    _sampleExponential(mean) {
        return SvendMath.sampleExponential(mean);
    }

    _sampleNormal(mean, cv) {
        return SvendMath.sampleNormal(mean, cv);
    }

    _sampleWeibull(eta, beta) {
        return SvendMath.sampleWeibull(eta, beta);
    }

    _sampleInterarrival(src) {
        if (src.arrivalDist === 'constant') return src.arrivalRate;
        if (src.arrivalDist === 'normal') return this._sampleNormal(src.arrivalRate, src.arrivalCV || 0.15);
        return this._sampleExponential(src.arrivalRate);
    }

    _sampleProcessTime(stn) {
        return this._sampleNormal(stn.cycleTime, stn.cycleTimeCV);
    }

    _updateStationTime(stn, newState) {
        const elapsed = this.clock - stn.stateStartTime;
        if (stn.stats[stn.state] !== undefined) stn.stats[stn.state] += elapsed;
        // Track when machine goes idle for warmup calculation
        if (newState === 'idle' && stn.state !== 'idle') stn.idleSince = this.clock;
        stn.state = newState;
        stn.stateStartTime = this.clock;
    }

    _getDownstream(fromId) {
        return this.adjacency.get(fromId) || [];
    }

    _getQueueLimit(fromId, toId) {
        const edges = this.adjacency.get(fromId) || [];
        const edge = edges.find(e => e.toId === toId);
        return edge ? edge.bufferCapacity : null;
    }

    _canAccept(stnId) {
        const stn = this.stations.get(stnId);
        if (!stn) return false;
        if (stn.type === 'sink') return true;
        // Check buffer capacity from upstream
        const upstreams = this.reverseAdj.get(stnId) || [];
        for (const upId of upstreams) {
            const limit = this._getQueueLimit(upId, stnId);
            if (limit !== null && limit !== undefined && stn.queue.length >= limit) return false;
        }
        return true;
    }

    processNextEvent() {
        if (this.eventQueue.size === 0) return false;
        const event = this.eventQueue.pop();
        this.clock = event.time;
        if (this.clock > this.warmupTime + this.runTime) return false;

        switch (event.type) {
            case 'arrival': this._handleArrival(event); break;
            case 'start_processing': this._handleStartProcessing(event); break;
            case 'end_processing': this._handleEndProcessing(event); break;
            case 'breakdown': this._handleBreakdown(event); break;
            case 'repair': this._handleRepair(event); break;
            case 'pm_start': this._handlePMStart(event); break;
            case 'pm_end': this._handlePMEnd(event); break;
            case 'break_start': this._handleBreakStart(event); break;
            case 'break_end': this._handleBreakEnd(event); break;
            case 'workforce_check': this._handleWorkforceCheck(); break;
            case 'training_complete': this._handleTrainingComplete(event); break;
            case 'changeover_end': this._handleChangeoverEnd(event); break;
            case 'transport_arrive': this._handleTransportArrive(event); break;
            case 'customer_demand': this._handleCustomerDemand(event); break;
            case 'schedule_review': this._handleScheduleReview(); break;
            case 'rush_check': this._handleRushCheck(); break;
            case 'early_leave': this._handleEarlyLeave(event); break;
            case 'operator_return': this._handleOperatorReturn(event); break;
            case 'record_stats': this._handleRecordStats(); break;
            case 'utility_failure': this._handleUtilityFailure(event); break;
            case 'utility_restore': this._handleUtilityRestore(event); break;
            case 'management_review': this._handleManagementReview(); break;
            case 'eco_event': this._handleECO(); break;
            case 'inspection_complete': this._handleInspectionComplete(event); break;
            case 'mission_event': this._handleMissionEvent(event); break;
        }
        return true;
    }

    _sampleProductType(src) {
        const types = src.productTypes || [{ name: 'A', ratio: 1.0, color: '#4a9f6e' }];

        // Use effective ratios if set (from dynamic scheduling)
        const getRatio = (t) => t._effectiveRatio != null ? t._effectiveRatio : (t.ratio || 0);
        const totalRatio = types.reduce((s, t) => s + getRatio(t), 0);
        if (totalRatio <= 0) return types[0]; // Fallback if all zero
        let r = Math.random() * totalRatio;
        for (const t of types) {
            r -= getRatio(t);
            if (r <= 0) return t;
        }
        return types[types.length - 1];
    }

    _getTransportTime(fromId, toId) {
        const edges = this.adjacency.get(fromId) || [];
        const edge = edges.find(e => e.toId === toId);
        if (!edge) return 0;

        const typeDef = TRANSPORT_TYPES[edge.transportType] || TRANSPORT_TYPES.none;
        if (typeDef.speed === Infinity || !edge.transportDistance) return 0;

        // Base time = distance / speed (distance in meters, speed in m/s)
        const baseTime = edge.transportDistance / typeDef.speed;

        // Add variability: walking is steady, forklifts have congestion/waits
        // CV: walk=0.1, pallet_jack=0.15, electric_pj=0.2, forklift=0.3, agv=0.1
        const cvMap = { walk: 0.1, hand_cart: 0.12, pallet_jack: 0.15, electric_pj: 0.2, forklift: 0.3, agv: 0.1 };
        const cv = cvMap[edge.transportType] || 0;
        const time = this._sampleNormal(baseTime, cv);

        // Breakdown: forklift/electric PJ might break down during transport
        if (typeDef.breakdownRate > 0 && Math.random() < typeDef.breakdownRate) {
            // Transport breakdown: add repair delay (2-10 min)
            const repairTime = 120 + Math.random() * 480;
            return time + repairTime;
        }

        return time;
    }

    _getChangeoverTime(stn, fromType, toType) {
        if (!fromType || !toType || fromType === toType) return 0;
        // Check setup matrix for sequence-dependent time
        const matrixKey = `${fromType}→${toType}`;
        if (stn.setupMatrix && stn.setupMatrix[matrixKey] != null) {
            return stn.setupMatrix[matrixKey];
        }
        // Fall back to flat changeover_time
        return stn.changeoverTime || 0;
    }

    _handleArrival(event) {
        const src = this.stations.get(event.stationId);
        if (!src) return;

        // If source is paused (kanban/chase — all FG above target), skip but re-check soon
        if (src._paused) {
            this.eventQueue.push({ time: this.clock + 30, type: 'arrival', stationId: src.id });
            return;
        }

        // Customer behavior: low satisfaction = lost orders (customers go elsewhere)
        if (this.customerConfig.revenuePerUnit > 0 && this.customerSatisfaction < 1.0) {
            // Below 50% satisfaction, start losing orders proportionally
            // At 50% sat → 0% loss, at 0% sat → 80% loss (sigmoid-ish)
            const lossRate = this.customerSatisfaction < 0.5
                ? (0.5 - this.customerSatisfaction) * 1.6  // max 80% at 0 sat
                : 0;
            if (lossRate > 0 && Math.random() < lossRate) {
                this.customerLostOrders++;
                // Still schedule next arrival (the market keeps ticking)
                let nextInterval = this._sampleInterarrival(src);
                if (src.supplierReliability < 1 && Math.random() > src.supplierReliability) {
                    nextInterval += src.lateDeliveryPenalty || (nextInterval * (0.5 + Math.random() * 1.5));
                }
                this.eventQueue.push({ time: this.clock + nextInterval, type: 'arrival', stationId: src.id });
                return;
            }
        }

        // Sample product type from source mix
        const pt = this._sampleProductType(src);

        // Batch sequence: decrement remaining
        if (src.scheduleMode === 'batch_sequence' && src.currentBatchRemaining > 0) {
            src.currentBatchRemaining--;
        }

        // Determine if this is a rush order
        const isRush = src.rushOrderRate > 0 && Math.random() < src.rushOrderRate;
        if (isRush) this.rushOrderCount++;

        // Create job
        const job = {
            id: ++this.jobIdCounter,
            createdAt: this.clock,
            productType: pt.name,
            productColor: pt.color || '#4a9f6e',
            stationId: null,
            completedAt: null,
            priority: isRush ? 'rush' : 'normal',   // normal | rush | hot
            dueDate: src.dueDateTarget > 0
                ? this.clock + (isRush ? src.dueDateTarget * 0.5 : src.dueDateTarget)
                : null,
            _materialQualityPenalty: 0,  // extra cycle time multiplier from bad material
            _expiresAt: pt.shelf_life > 0 ? this.clock + pt.shelf_life : null,  // WIP aging
        };

        // Track material cost
        if (src.materialCostPerUnit > 0) {
            this.costs.material += src.materialCostPerUnit;
            job._materialCost = src.materialCostPerUnit;
        }

        // SUPPLIER VARIABILITY: incoming material quality
        if (src.incomingQualityRate < 1 && Math.random() > src.incomingQualityRate) {
            // Bad material batch: job carries a hidden processing penalty
            // Bad material = 10-30% slower processing + 2× scrap risk at each station
            job._materialQualityPenalty = 0.1 + Math.random() * 0.2;
            job._badMaterial = true;
        }
        this.jobs.push(job);

        // Route to downstream (with transport time)
        const downstream = this._getDownstream(src.id);
        if (downstream.length > 0) {
            const target = this._pickTarget(downstream, job);
            const transportTime = this._getTransportTime(src.id, target.toId);
            if (transportTime > 0) {
                this._requestTransport(job, target.toId, transportTime);
            } else {
                this._deliverJob(job, target.toId);
            }
        }

        // Schedule next arrival (supplier reliability affects interarrival)
        let nextInterval = this._sampleInterarrival(src);
        // Supplier late delivery: some arrivals take much longer
        if (src.supplierReliability < 1 && Math.random() > src.supplierReliability) {
            nextInterval += src.lateDeliveryPenalty || (nextInterval * (0.5 + Math.random() * 1.5));
        }
        const nextTime = this.clock + nextInterval;
        this.eventQueue.push({ time: nextTime, type: 'arrival', stationId: src.id });
    }

    _pickTarget(downstream, job = null) {
        if (downstream.length === 1) {
            const d = downstream[0];
            // Routing rules filter (CR-4): if rules exist, product must be listed
            if (d.routingRules?.length > 0 && job) {
                const rule = d.routingRules.find(r => r.product === job.productType);
                if (!rule) return null;
                if (rule.weight < 1.0 && Math.random() > rule.weight) return null;
            }
            // Dedicated machine filter (CR-2)
            const stn = this.stations.get(d.toId);
            if (stn && stn.dedicatedProduct && job && job.productType !== stn.dedicatedProduct) return null;
            return d;
        }
        // Pick eligible station with shortest queue
        let best = null;
        let bestLen = Infinity;
        for (const d of downstream) {
            // Routing rules filter (CR-4): if rules exist, product must be listed
            if (d.routingRules?.length > 0 && job) {
                const rule = d.routingRules.find(r => r.product === job.productType);
                if (!rule) continue;
                if (rule.weight < 1.0 && Math.random() > rule.weight) continue;
            }
            const stn = this.stations.get(d.toId);
            if (!stn) continue;
            // Dedicated machine filter (CR-2)
            if (stn.dedicatedProduct && job && job.productType !== stn.dedicatedProduct) continue;
            const len = stn.queue.length;
            if (len < bestLen) { bestLen = len; best = d; }
        }
        return best || downstream[0]; // fallback if no match
    }

    _deliverJob(job, toId) {
        const targetStn = this.stations.get(toId);
        if (!targetStn) return;
        if (targetStn.type === 'sink') {
            job.completedAt = this.clock;
            job.stationId = targetStn.id;
            // Track on-time vs late delivery
            if (job.dueDate != null) {
                if (this.clock <= job.dueDate) this.onTimeCount++;
                else this.lateOrderCount++;
            }
            this._recordCustomerDelivery(job);
            this.completedJobs.push(job);
            // FG warehouse mode: track inventory by product type
            if (targetStn.sinkMode === 'fg_warehouse') {
                const pt = job.productType || 'A';
                targetStn.fgInventory[pt] = (targetStn.fgInventory[pt] || 0) + 1;
            }
        } else if (targetStn.type === 'machine') {
            // Assembly mode: collect parts in buffer until all inputs are met
            if (targetStn.assemblyMode && Object.keys(targetStn.assemblyInputs).length > 0) {
                this._assemblyReceive(targetStn, job);
            } else {
                this._enqueueJob(targetStn, job);
                job.stationId = targetStn.id;
                this._tryStartProcessing(targetStn.id);
            }
        }
    }

    // Priority-aware queue insertion: rush/hot orders jump ahead of normal orders
    _enqueueJob(stn, job) {
        if (job.priority === 'normal' || !job.priority) {
            stn.queue.push(job);
            return;
        }
        // Rush/hot: insert before the first normal-priority job
        const insertIdx = stn.queue.findIndex(j => j.priority === 'normal' || !j.priority);
        if (insertIdx === -1) {
            stn.queue.push(job); // all jobs are rush/hot, add at end
        } else {
            stn.queue.splice(insertIdx, 0, job);
        }
    }

    // SPC: Western Electric rules for control chart signals
    // Returns true if any rule is violated
    _checkSPCRules(measurements) {
        const n = measurements.length;
        if (n < 8) return false; // need minimum data

        const sigma = 3.0; // control limit at 3σ

        // Rule 1: One point beyond 3σ
        const last = measurements[n - 1];
        if (Math.abs(last) > sigma) return true;

        // Rule 2: 2 of 3 consecutive points beyond 2σ (same side)
        if (n >= 3) {
            const recent3 = measurements.slice(-3);
            const above2 = recent3.filter(v => v > sigma * 2/3).length;
            const below2 = recent3.filter(v => v < -sigma * 2/3).length;
            if (above2 >= 2 || below2 >= 2) return true;
        }

        // Rule 3: 4 of 5 consecutive points beyond 1σ (same side)
        if (n >= 5) {
            const recent5 = measurements.slice(-5);
            const above1 = recent5.filter(v => v > sigma * 1/3).length;
            const below1 = recent5.filter(v => v < -sigma * 1/3).length;
            if (above1 >= 4 || below1 >= 4) return true;
        }

        // Rule 4: 8 consecutive points on one side of center
        if (n >= 8) {
            const recent8 = measurements.slice(-8);
            if (recent8.every(v => v > 0) || recent8.every(v => v < 0)) return true;
        }

        return false;
    }

    // Customer behavior: record a completed delivery and update satisfaction/revenue
    _recordCustomerDelivery(job) {
        const cc = this.customerConfig;
        if (cc.revenuePerUnit <= 0) return; // disabled

        const isReturn = job._hasDefect || job._contaminationDefect;
        const isLate = job.dueDate != null && this.clock > job.dueDate;

        if (isReturn) {
            // Return costs: warranty + shipping + replacement + goodwill damage
            const returnCost = cc.revenuePerUnit * cc.returnCostMultiplier;
            this.totalReturnCost += returnCost;
            // Satisfaction hit: each return is a -3% hit (compounds badly)
            this.customerSatisfaction = Math.max(0, this.customerSatisfaction - 0.03);
        } else {
            // Good delivery: revenue earned
            this.totalRevenue += cc.revenuePerUnit;
            // Late but not defective: smaller satisfaction hit
            if (isLate) {
                this.customerSatisfaction = Math.max(0, this.customerSatisfaction - 0.01);
            } else {
                // On-time good delivery: slow satisfaction recovery
                this.customerSatisfaction = Math.min(1, this.customerSatisfaction + 0.002);
            }
        }
    }

    // Assembly station: receive a part into the assembly buffer
    _assemblyReceive(stn, job) {
        const pt = job.productType || 'A';
        if (!stn.assemblyBuffer[pt]) stn.assemblyBuffer[pt] = [];
        stn.assemblyBuffer[pt].push(job);
        job.stationId = stn.id;
        this._assemblyCheck(stn);
    }

    // Check if all assembly inputs are satisfied; if so, create a composite job
    _assemblyCheck(stn) {
        const inputs = stn.assemblyInputs;
        // Check each required input
        for (const [pt, qty] of Object.entries(inputs)) {
            if (!stn.assemblyBuffer[pt] || stn.assemblyBuffer[pt].length < qty) return; // not ready
        }
        // All inputs met — consume parts and create assembled job
        let earliestCreated = 0;
        let highestPriority = 'normal';
        let earliestDue = null;
        let totalMaterialCost = 0;
        for (const [pt, qty] of Object.entries(inputs)) {
            for (let i = 0; i < qty; i++) {
                const part = stn.assemblyBuffer[pt].shift();
                if (part.createdAt > earliestCreated) earliestCreated = part.createdAt;
                if (part.priority === 'hot' || (part.priority === 'rush' && highestPriority !== 'hot')) {
                    highestPriority = part.priority;
                }
                if (part.dueDate && (!earliestDue || part.dueDate < earliestDue)) {
                    earliestDue = part.dueDate;
                }
                totalMaterialCost += (part._materialCost || 0);
            }
        }
        // Create the assembled job
        const assembledJob = {
            id: ++this.jobIdCounter,
            createdAt: earliestCreated,  // lead time includes wait for all parts
            productType: stn.name || 'Assembly',
            productColor: '#9b59b6',  // purple for assembled items
            stationId: stn.id,
            completedAt: null,
            priority: highestPriority,
            dueDate: earliestDue,
            _materialQualityPenalty: 0,
            _materialCost: totalMaterialCost,
            _expiresAt: null,
            _assembled: true,
        };
        this.jobs.push(assembledJob);
        this._enqueueJob(stn, assembledJob);
        this._tryStartProcessing(stn.id);
    }

    // Dispatch rule sorting — reorders queue within priority tiers
    // FIFO: arrival order (default, no sort needed)
    // SPT: shortest processing time first (minimizes avg lead time)
    // EDD: earliest due date first (minimizes max lateness)
    // CR: critical ratio = time remaining / work remaining (lowest first, <1 means behind schedule)
    // WSPT: weighted shortest processing time = priority_weight / processing_time (highest first)
    _sortByDispatchRule(stn) {
        const rule = stn.dispatchRule || 'FIFO';
        if (rule === 'FIFO' || stn.queue.length <= 1) return;

        const priorityRank = (j) => (j.priority === 'hot' ? 0 : j.priority === 'rush' ? 1 : 2);
        const estProcTime = (j) => {
            // Estimate based on station cycle time (actual will vary with skill/fatigue)
            return stn.cycleTime || 30;
        };

        const comparator = (a, b) => {
            // Always respect priority tiers first
            const pa = priorityRank(a), pb = priorityRank(b);
            if (pa !== pb) return pa - pb;

            switch (rule) {
                case 'SPT':
                    return estProcTime(a) - estProcTime(b);
                case 'EDD':
                    // Jobs without due dates go to the back
                    if (!a.dueDate && !b.dueDate) return 0;
                    if (!a.dueDate) return 1;
                    if (!b.dueDate) return -1;
                    return a.dueDate - b.dueDate;
                case 'CR': {
                    // Critical Ratio = (due date - now) / estimated remaining work
                    // Lower CR = more urgent. No due date = CR of Infinity (goes last)
                    const crA = a.dueDate ? (a.dueDate - this.clock) / Math.max(estProcTime(a), 1) : Infinity;
                    const crB = b.dueDate ? (b.dueDate - this.clock) / Math.max(estProcTime(b), 1) : Infinity;
                    return crA - crB;
                }
                case 'WSPT': {
                    // Weighted SPT: weight / processing time (higher = higher priority, so sort descending)
                    const wA = (a.priority === 'rush' || a.priority === 'hot') ? 2 : 1;
                    const wB = (b.priority === 'rush' || b.priority === 'hot') ? 2 : 1;
                    return (wB / estProcTime(b)) - (wA / estProcTime(a));
                }
                default:
                    return 0;
            }
        };

        stn.queue.sort(comparator);
    }

    _handleTransportArrive(event) {
        this._deliverJob(event.job, event.toId);
        // Release AGV back to pool
        if (this.agvFleet.size > 0 && event._agvTrip) {
            this.agvFleet.available++;
            this.agvFleet.tripsCompleted++;
            this._dispatchNextTransport();
        }
    }

    // AGV fleet: request a transport vehicle
    _requestTransport(job, toId, transportTime) {
        const fleet = this.agvFleet;
        if (fleet.size === 0) {
            // No fleet — transport is instantaneous (handled by caller)
            this.eventQueue.push({ time: this.clock + transportTime, type: 'transport_arrive', job, toId });
            return;
        }
        if (fleet.available > 0) {
            fleet.available--;
            this.eventQueue.push({ time: this.clock + transportTime, type: 'transport_arrive', job, toId, _agvTrip: true });
        } else {
            fleet.transportQueue.push({ job, toId, transportTime, requestedAt: this.clock });
        }
    }

    _dispatchNextTransport() {
        const fleet = this.agvFleet;
        if (fleet.available <= 0 || fleet.transportQueue.length === 0) return;
        const next = fleet.transportQueue.shift();
        fleet.available--;
        fleet.totalWaitTime += (this.clock - next.requestedAt);
        this.eventQueue.push({ time: this.clock + next.transportTime, type: 'transport_arrive', job: next.job, toId: next.toId, _agvTrip: true });
    }

    _handleCustomerDemand(event) {
        const sink = this.stations.get(event.stationId);
        if (!sink || sink.type !== 'sink' || sink.sinkMode !== 'fg_warehouse') return;

        const product = event.product;
        const available = sink.fgInventory[product] || 0;

        if (available > 0) {
            // Fill demand from FG
            sink.fgInventory[product] = available - 1;
            sink.demandFilled[product] = (sink.demandFilled[product] || 0) + 1;
        } else {
            // Stockout — customer demand unfilled
            sink.stockouts[product] = (sink.stockouts[product] || 0) + 1;
        }

        // Schedule next demand for this product
        const dr = sink.demandRates.find(d => d.product === product);
        if (dr && dr.rate > 0) {
            const interarrival = 3600 / dr.rate;
            let nextTime;
            if (dr.distribution === 'constant') {
                nextTime = interarrival;
            } else if (dr.distribution === 'normal') {
                nextTime = this._sampleNormal(interarrival, dr.cv || 0.2);
            } else {
                nextTime = this._sampleExponential(interarrival);
            }
            this.eventQueue.push({
                time: this.clock + nextTime,
                type: 'customer_demand',
                stationId: event.stationId,
                product,
            });
        }
    }

    _handleScheduleReview() {
        // Dynamic scheduling: check FG levels and adjust source production
        for (const [id, src] of this.stations) {
            if (src.type !== 'source') continue;
            if (src.scheduleMode === 'fixed_mix') continue;

            // Find connected FG warehouse(s) by tracing downstream through machines to sinks
            const fgSinks = this._findDownstreamSinks(id);
            if (fgSinks.length === 0) continue;

            // Aggregate FG inventory and demand across all connected sinks
            const fgLevels = {};
            const targets = {};  // safety stock or reorder point
            for (const sinkId of fgSinks) {
                const sink = this.stations.get(sinkId);
                if (!sink || sink.sinkMode !== 'fg_warehouse') continue;
                for (const pt of src.productTypes) {
                    const pn = pt.name;
                    fgLevels[pn] = (fgLevels[pn] || 0) + (sink.fgInventory[pn] || 0);
                    if (src.scheduleMode === 'kanban') {
                        targets[pn] = Math.max(targets[pn] || 0, sink.reorderPoint[pn] || 20);
                    } else {
                        targets[pn] = Math.max(targets[pn] || 0, sink.safetyStock[pn] || 30);
                    }
                }
            }

            if (src.scheduleMode === 'kanban') {
                // Kanban: produce whatever is below reorder point, prioritize lowest relative level
                let worstProduct = null;
                let worstDeficit = 0;
                for (const pt of src.productTypes) {
                    const level = fgLevels[pt.name] || 0;
                    const target = targets[pt.name] || 20;
                    const deficit = target - level;
                    if (deficit > worstDeficit) {
                        worstDeficit = deficit;
                        worstProduct = pt.name;
                    }
                }
                // Set all ratios to 0 except the one we need most
                if (worstProduct) {
                    for (const pt of src.productTypes) {
                        pt._effectiveRatio = pt.name === worstProduct ? 1.0 : 0;
                    }
                } else {
                    // All above reorder point — stop production
                    for (const pt of src.productTypes) {
                        pt._effectiveRatio = 0;
                    }
                    src._paused = true;
                }
                if (worstDeficit > 0) src._paused = false;
            } else if (src.scheduleMode === 'chase_demand') {
                // Chase: adjust ratios proportional to deficit from safety stock
                let totalDeficit = 0;
                const deficits = {};
                for (const pt of src.productTypes) {
                    const level = fgLevels[pt.name] || 0;
                    const target = targets[pt.name] || 30;
                    const deficit = Math.max(0, target - level);
                    deficits[pt.name] = deficit;
                    totalDeficit += deficit;
                }
                for (const pt of src.productTypes) {
                    pt._effectiveRatio = totalDeficit > 0 ? (deficits[pt.name] / totalDeficit) : pt.ratio;
                }
                src._paused = totalDeficit === 0;
            } else if (src.scheduleMode === 'batch_sequence') {
                // Batch: run one product type at a time, switch when batch complete or FG full
                if (src.currentBatchRemaining <= 0) {
                    // Pick next product: lowest FG relative to safety stock
                    let worstProduct = src.productTypes[0]?.name;
                    let worstRatio = Infinity;
                    for (const pt of src.productTypes) {
                        const level = fgLevels[pt.name] || 0;
                        const target = targets[pt.name] || 30;
                        const ratio = level / Math.max(1, target);
                        if (ratio < worstRatio) {
                            worstRatio = ratio;
                            worstProduct = pt.name;
                        }
                    }
                    src.currentBatchType = worstProduct;
                    src.currentBatchRemaining = 50; // Default batch size
                }
                for (const pt of src.productTypes) {
                    pt._effectiveRatio = pt.name === src.currentBatchType ? 1.0 : 0;
                }
            }
        }

        // Schedule next review
        this.eventQueue.push({ time: this.clock + 300, type: 'schedule_review' });
    }

    _handleRushCheck() {
        // THE EXPEDITING DEATH SPIRAL
        // Scan all WIP: if a job has consumed too much of its due date budget, promote it to rush.
        // Rush jobs jump queues → cause extra changeovers → delay other jobs → more jobs go late → more rush.
        let promoted = 0;
        for (const [id, stn] of this.stations) {
            if (stn.type !== 'machine') continue;
            for (const job of stn.queue) {
                if (job.priority !== 'normal' || !job.dueDate) continue;
                const elapsed = this.clock - job.createdAt;
                const budget = job.dueDate - job.createdAt;
                if (budget > 0 && (elapsed / budget) > 0.8) {
                    // This job is dangerously late — promote to rush
                    job.priority = 'rush';
                    job.productColor = '#ff6b6b'; // Visual: turn red
                    promoted++;
                    this.promotedToRush++;
                }
            }
            // Also check current job being processed
            if (stn.currentJob && stn.currentJob.priority === 'normal' && stn.currentJob.dueDate) {
                const elapsed = this.clock - stn.currentJob.createdAt;
                const budget = stn.currentJob.dueDate - stn.currentJob.createdAt;
                if (budget > 0 && (elapsed / budget) > 0.9) {
                    stn.currentJob.priority = 'rush'; // can't reorder mid-process, but track it
                    this.promotedToRush++;
                }
            }
            // Re-sort queue by priority after promotions
            if (promoted > 0) {
                const rushJobs = stn.queue.filter(j => j.priority === 'rush' || j.priority === 'hot');
                const normalJobs = stn.queue.filter(j => j.priority === 'normal' || !j.priority);
                stn.queue = [...rushJobs, ...normalJobs];
            }
        }

        // Schedule next check
        this.eventQueue.push({ time: this.clock + 60, type: 'rush_check' });
    }

    _findDownstreamSinks(sourceId) {
        // BFS from source through all downstream connections to find sinks
        const visited = new Set();
        const queue = [sourceId];
        const sinks = [];
        while (queue.length > 0) {
            const current = queue.shift();
            if (visited.has(current)) continue;
            visited.add(current);
            const stn = this.stations.get(current);
            if (stn && stn.type === 'sink') sinks.push(current);
            const downstream = this.adjacency.get(current) || [];
            for (const d of downstream) queue.push(d.toId);
        }
        return sinks;
    }

    _findOperator(stnId) {
        if (this.operators.length === 0) return null; // No operator system — skip
        // Find available operator with best skill for this station
        let best = null;
        let bestSkill = -1;
        for (const op of this.operators) {
            if (op.status !== 'available') continue;
            const skill = op.skills[stnId] ?? 0;
            if (skill > bestSkill) { bestSkill = skill; best = op; }
        }
        return best;
    }

    _assignOperator(op, stnId) {
        op.status = 'busy';
        op.assignedTo = stnId;
    }

    _releaseOperator(stnId) {
        const op = this.operators.find(o => o.assignedTo === stnId && o.status === 'busy');
        if (op) {
            op.status = 'available';
            op.assignedTo = null;
            op.jobsThisShift++;
            // Skill improvement: small gain each job completed (learning by doing)
            const current = op.skills[stnId] ?? 0;
            if (current < 1) {
                op.skills[stnId] = Math.min(1, current + 0.002); // ~500 jobs to master
            }
            op.lastWorkedAt[stnId] = this.clock; // stamp for forgetting curve
            // Tribal knowledge: slowly accumulate process tweaks (much slower than skill)
            const tweak = op.processTweaks[stnId] ?? 0;
            if (tweak < 1) {
                op.processTweaks[stnId] = Math.min(1, tweak + 0.0005); // ~2000 jobs to master tricks
            }
            // Fatigue accumulation: grows over shift, faster when morale is low
            const hoursIntoShift = (this.clock - op.shiftStart) / 3600;
            // Fatigue curve: slow first 4hrs, accelerates after. Low morale = faster fatigue.
            const moraleMultiplier = 1 + (1 - op.morale) * 0.5;
            op.fatigue = Math.min(1, (hoursIntoShift / 8) * 0.7 * moraleMultiplier);
            // After 8hrs (overtime), fatigue climbs steeply
            if (hoursIntoShift > 8) {
                const otHours = hoursIntoShift - 8;
                op.fatigue = Math.min(1, op.fatigue + otHours * 0.15);
                op.overtimeHours = otHours;
            }
        }
        // Try to start other starving machines now that an operator is free
        for (const [id, s] of this.stations) {
            if (s.type === 'machine' && s.state === 'starved' && s.queue.length > 0) {
                this._tryStartProcessing(id);
            }
        }
    }

    _tryStartProcessing(stnId) {
        const stn = this.stations.get(stnId);
        if (!stn || stn.type !== 'machine') return;
        if (stn._manualStop) return;
        if (stn.state === 'processing' || stn.state === 'setup' || stn.state === 'down' || stn.state === 'onBreak') return;
        if (stn.queue.length === 0) {
            if (stn.state !== 'starved' && stn.state !== 'idle') this._updateStationTime(stn, 'starved');
            return;
        }

        // Dedicated machine: only process matching product type
        if (stn.dedicatedProduct) {
            const matchIdx = stn.queue.findIndex(j => j.productType === stn.dedicatedProduct);
            if (matchIdx === -1) {
                // No matching jobs in queue — starve
                if (stn.state !== 'starved') this._updateStationTime(stn, 'starved');
                return;
            }
            // Move matching job to front of queue
            if (matchIdx > 0) {
                const [matching] = stn.queue.splice(matchIdx, 1);
                stn.queue.unshift(matching);
            }
        }

        // Check if blocked (downstream full)
        const downstream = this._getDownstream(stn.id);
        if (downstream.length > 0 && stn.blockedJob) {
            return;
        }

        // Check operator availability (only if operators exist in the sim)
        if (this.operators.length > 0) {
            const op = this._findOperator(stn.id);
            if (!op) {
                // No operator available — machine starves waiting for one
                if (stn.state !== 'starved') this._updateStationTime(stn, 'starved');
                return;
            }
        }

        // Check shared tooling: does this machine need a tool that's unavailable?
        if (this.sharedTools.length > 0) {
            const needed = this.sharedTools.find(t => t.requiredBy.includes(stn.id));
            if (needed && needed.available <= 0) {
                // Tool not available — queue for it
                if (!needed.toolQueue.find(q => q.stationId === stn.id)) {
                    needed.toolQueue.push({ stationId: stn.id, requestedAt: this.clock });
                }
                if (stn.state !== 'starved') this._updateStationTime(stn, 'starved');
                return;
            }
            // Acquire tool
            if (needed) {
                needed.available--;
                stn._holdingTool = needed.id;
            }
        }

        this.eventQueue.push({ time: this.clock, type: 'start_processing', stationId: stnId });
    }

    _handleStartProcessing(event) {
        const stn = this.stations.get(event.stationId);
        if (!stn || stn.queue.length === 0) return;
        if (stn.state === 'processing' || stn.state === 'down' || stn.state === 'setup') return;

        // Check for changeover: product type differs from last
        const nextJob = stn.queue[0];
        if (nextJob && nextJob.productType && stn.lastProductType && nextJob.productType !== stn.lastProductType) {
            let coTime = this._getChangeoverTime(stn, stn.lastProductType, nextJob.productType);
            if (coTime > 0) {
                stn.stats.changeovers++;
                // Contamination: quick changeover may leave residue
                if (stn.contaminationRisk > 0) {
                    // Standard changeover = quick = contamination risk
                    // Machine has contamination_risk chance of being contaminated after quick changeover
                    stn._contaminated = Math.random() < stn.contaminationRisk;
                    if (stn._contaminated) stn.contaminationEvents++;
                }
                stn.lastProductType = nextJob.productType;
                this._updateStationTime(stn, 'setup');
                this.eventQueue.push({
                    time: this.clock + coTime,
                    type: 'changeover_end',
                    stationId: stn.id,
                });
                return;
            }
        }

        // Batch processing mode (oven/furnace): wait for N parts, process all at once
        if (stn.batchProcessMode && stn.queue.length < stn.batchProcessSize) {
            // Not enough parts yet — stay idle/starved waiting for a full batch
            if (stn.state !== 'starved') this._updateStationTime(stn, 'starved');
            return;
        }
        if (stn.batchProcessMode && stn.queue.length >= stn.batchProcessSize) {
            // Take the whole batch
            const batch = stn.queue.splice(0, stn.batchProcessSize);
            stn.currentJob = batch[0]; // representative job
            stn.currentJob._batchJobs = batch; // stash all batch jobs
            stn.lastProductType = batch[0].productType;
            this._updateStationTime(stn, 'processing');
            this.eventQueue.push({
                time: this.clock + stn.batchProcessTime,
                type: 'end_processing',
                stationId: stn.id,
            });
            return;
        }

        // Acquire operator
        let skillFactor = 1;
        let fatigueFactor = 1;
        let tweakFactor = 1;  // tribal knowledge: process tricks reduce cycle time
        if (this.operators.length > 0) {
            const op = this._findOperator(stn.id);
            if (!op) {
                // Operator became unavailable between tryStart and now
                if (stn.state !== 'starved') this._updateStationTime(stn, 'starved');
                return;
            }
            this._assignOperator(op, stn.id);
            const skill = op.skills[stn.id] ?? 0;
            // Skill penalty: untrained operator runs slower
            // skill=1.0 → factor=1.0, skill=0.3 → factor=1.35, skill=0 → factor=1.5
            skillFactor = 1 + (1 - skill) * 0.5;
            // Fatigue penalty: tired operators run 5-20% slower
            fatigueFactor = 1 + op.fatigue * 0.2;
            // Tribal knowledge: accumulated process tweaks improve speed by up to 10%
            const tweakLevel = op.processTweaks[stn.id] ?? 0;
            tweakFactor = 1 - tweakLevel * 0.1;  // tweak=1.0 → 0.9 (10% faster)
        }

        // Apply dispatch rule to sort queue (within priority tiers)
        this._sortByDispatchRule(stn);

        // WIP aging: discard expired jobs from front of queue
        while (stn.queue.length > 0 && stn.queue[0]._expiresAt && this.clock > stn.queue[0]._expiresAt) {
            const expired = stn.queue.shift();
            this.expiredWIP++;
            stn.stats.scrapped++;
            this.costs.scrapWaste += (expired._materialCost || 0);
        }
        if (stn.queue.length === 0) {
            if (this.operators.length > 0) {
                // Release the operator we just acquired
                const op = this.operators.find(o => o.assignedTo === stn.id && o.status === 'busy');
                if (op) { op.status = 'available'; op.assignedTo = null; }
            }
            if (stn.state !== 'starved') this._updateStationTime(stn, 'starved');
            return;
        }

        stn.currentJob = stn.queue.shift();
        stn.lastProductType = stn.currentJob.productType || null;
        // Material quality penalty: bad incoming material slows processing
        const materialFactor = 1 + (stn.currentJob._materialQualityPenalty || 0);
        let processTime = this._sampleProcessTime(stn) * skillFactor * fatigueFactor * materialFactor * tweakFactor;
        // Micro-stoppages: hidden time loss that inflates cycle time
        if (stn.microStopRate > 0 && Math.random() < stn.microStopRate) {
            const stopTime = this._sampleExponential(stn.microStopDuration);
            processTime += stopTime;
            stn.microStopCount++;
            stn.microStopLostTime += stopTime;
        }
        this._updateStationTime(stn, 'processing');

        this.eventQueue.push({
            time: this.clock + processTime,
            type: 'end_processing',
            stationId: stn.id,
        });
    }

    _handleChangeoverEnd(event) {
        const stn = this.stations.get(event.stationId);
        if (!stn) return;
        // Changeover done — proceed to process the next job
        this._updateStationTime(stn, 'idle');
        this._tryStartProcessing(stn.id);
    }

    _handleEndProcessing(event) {
        const stn = this.stations.get(event.stationId);
        if (!stn || !stn.currentJob) return;

        // Release operator — they're free regardless of outcome
        this._releaseOperator(stn.id);

        // Release shared tool and dispatch next queued machine
        this._releaseSharedTool(stn);

        const job = stn.currentJob;

        // Batch processing: release all batch jobs downstream
        if (job._batchJobs && job._batchJobs.length > 1) {
            const batchJobs = job._batchJobs;
            stn.currentJob = null;
            stn.stats.processed += batchJobs.length;
            this._updateStationTime(stn, 'idle');
            const downstream = this._getDownstream(stn.id);
            for (const bj of batchJobs) {
                bj._batchJobs = null; // clean up
                if (downstream.length > 0) {
                    const target = this._pickTarget(downstream, bj);
                    const targetStn = this.stations.get(target.toId);
                    const transportTime = this._getTransportTime(stn.id, target.toId);
                    if (targetStn && targetStn.type === 'sink') {
                        if (bj._hasDefect || bj._contaminationDefect) this.customerReturns++;
                        this._recordCustomerDelivery(bj);
                        bj.completedAt = this.clock;
                        bj.stationId = targetStn.id;
                        this.completedJobs.push(bj);
                    } else if (transportTime > 0) {
                        this._requestTransport(bj, target.toId, transportTime);
                    } else {
                        this._deliverJob(bj, target.toId);
                    }
                } else {
                    if (bj._hasDefect || bj._contaminationDefect) this.customerReturns++;
                    this._recordCustomerDelivery(bj);
                    bj.completedAt = this.clock;
                    if (bj.dueDate != null) {
                        if (this.clock <= bj.dueDate) this.onTimeCount++;
                        else this.lateOrderCount++;
                    }
                    this.completedJobs.push(bj);
                }
            }
            this._tryStartProcessing(stn.id);
            return;
        }

        // Track operating time for Weibull degradation
        const processedTime = this.clock - stn.stateStartTime;
        stn.operatingTime += processedTime;

        // Process drift: scrap rate creeps up between calibrations
        stn.jobsSinceCalibration++;
        if (stn.driftRate > 0) {
            stn.driftAccumulated = stn.driftRate * (stn.jobsSinceCalibration / 100);
            // Auto-calibrate if interval set
            if (stn.calibrationInterval > 0 && stn.jobsSinceCalibration >= stn.calibrationInterval) {
                stn.driftAccumulated = 0;
                stn.jobsSinceCalibration = 0;
            }
        }

        // Track first-article status
        stn.partsAfterRestart++;

        // Per-product quality rates (CR-3): override flat rates if product-specific rates exist
        const _qbp = stn.qualityByProduct || {};
        const _pq = (job && _qbp[job.productType]) || {};
        const baseScrap = _pq.scrap_rate ?? stn.scrapRate;
        const baseDefect = _pq.defect_rate ?? (stn.defectRate || 0);
        const baseRework = _pq.rework_rate ?? stn.reworkRate;

        // Inspector bottleneck: if pool is active, request an inspector
        // If no inspector is available, job queues (machine releases but job waits)
        if (this.inspectorPool.size > 0 && (baseScrap + baseDefect + baseRework) > 0) {
            stn.stats.processed++;
            const jobToInspect = stn.currentJob;
            stn.currentJob = null;
            this._updateStationTime(stn, 'idle');
            this._tryStartProcessing(stn.id); // machine is free immediately
            // Stash quality-relevant state on the job for deferred inspection
            jobToInspect._inspectionStation = stn.id;
            jobToInspect._effectiveScrapRate = baseScrap + stn.driftAccumulated;
            jobToInspect._effectiveDefectRate = baseDefect;
            jobToInspect._effectiveReworkRate = baseRework;
            jobToInspect._defectDetectionRate = stn.defectDetectionRate;
            this._requestInspection(jobToInspect, stn.id);
            return;
        }

        // Quality check: scrap, defect, or rework (fatigue + drift + first-article increase error rates)
        const fatigueErrorBoost = this.operators.length > 0
            ? (this.operators.find(o => o.id && o.jobsThisShift > 0)?.fatigue || 0) * 0.5
            : 0;
        // First-article penalty: elevated scrap on first N parts after restart/idle
        const firstArticleBoost = (stn.firstArticlePenalty > 0 && stn.partsAfterRestart <= stn.firstArticleCount)
            ? stn.firstArticlePenalty : 0;
        // Bad material doubles the effective scrap contribution
        const materialScrapBoost = job._badMaterial ? (baseScrap + stn.driftAccumulated) : 0;
        // Tribal knowledge: process tweaks reduce scrap (experienced operator knows the tricks)
        const assignedOp = this.operators.find(o => o.assignedTo === stn.id && o.status === 'busy')
            || this.operators.find(o => o.lastWorkedAt[stn.id] === this.clock); // just released
        const tweakQualityBoost = assignedOp ? (assignedOp.processTweaks[stn.id] ?? 0) * 0.3 : 0; // up to 30% scrap reduction
        // Contamination: residue from previous product causes hidden quality issues
        const contaminationBoost = stn._contaminated ? 0.05 + Math.random() * 0.1 : 0; // 5-15% extra scrap
        if (stn._contaminated && Math.random() < 0.3) {
            // Contaminated part that passes inspection but fails at customer
            job._contaminationDefect = true;
        }
        const driftedScrap = baseScrap + stn.driftAccumulated + firstArticleBoost + materialScrapBoost + contaminationBoost;
        const effectiveScrap = Math.min(0.5, driftedScrap * (1 + fatigueErrorBoost) * (1 - tweakQualityBoost));
        const effectiveDefect = Math.min(0.5, baseDefect * (1 + fatigueErrorBoost) * (1 - tweakQualityBoost));
        const effectiveRework = Math.min(0.5, baseRework * (1 + fatigueErrorBoost) * (1 - tweakQualityBoost));

        if (baseDefect > 0) {
            // CR-3: 3-stage quality model (scrap ≠ defects)
            // Stage 1: Pure scrap (material waste — independent of defects)
            if (Math.random() < effectiveScrap) {
                stn.stats.scrapped++;
                stn.stats.processed++;
                this.costs.scrapWaste += (job._materialCost || 0);
                stn.currentJob = null;
                this._updateStationTime(stn, 'idle');
                this._tryStartProcessing(stn.id);
                return;
            }
            // Stage 2: Defect introduction (nonconformance)
            if (Math.random() < effectiveDefect) {
                const detected = Math.random() < stn.defectDetectionRate;
                if (detected) {
                    // Detected defect: reworkable or scrap?
                    if (effectiveRework > 0 && Math.random() < effectiveRework) {
                        stn.stats.reworked++;
                        stn.stats.processed++;
                        stn.currentJob = null;
                        if (stn.reworkTarget && this.stations.has(stn.reworkTarget)) {
                            this._deliverJob(job, stn.reworkTarget);
                        } else {
                            this._enqueueJob(stn, job);
                        }
                        this._updateStationTime(stn, 'idle');
                        this._tryStartProcessing(stn.id);
                        return;
                    } else {
                        // Detected defect, not reworkable → scrap
                        stn.stats.scrapped++;
                        stn.stats.processed++;
                        this.costs.scrapWaste += (job._materialCost || 0);
                        stn.currentJob = null;
                        this._updateStationTime(stn, 'idle');
                        this._tryStartProcessing(stn.id);
                        return;
                    }
                } else {
                    // ESCAPED DEFECT — job continues downstream carrying a hidden defect
                    this.escapedDefects++;
                    job._hasDefect = true;
                }
            }
        } else {
            // Legacy behavior (defect_rate = 0): single roll for scrap + rework
            const roll = Math.random();
            const isDefective = roll < effectiveScrap + effectiveRework;
            const isScrap = roll < effectiveScrap;

            if (isDefective) {
                const detected = Math.random() < stn.defectDetectionRate;
                if (detected) {
                    if (isScrap) {
                        stn.stats.scrapped++;
                        stn.stats.processed++;
                        this.costs.scrapWaste += (job._materialCost || 0);
                        stn.currentJob = null;
                        this._updateStationTime(stn, 'idle');
                        this._tryStartProcessing(stn.id);
                        return;
                    } else {
                        stn.stats.reworked++;
                        stn.stats.processed++;
                        stn.currentJob = null;
                        if (stn.reworkTarget && this.stations.has(stn.reworkTarget)) {
                            this._deliverJob(job, stn.reworkTarget);
                        } else {
                            this._enqueueJob(stn, job);
                        }
                        this._updateStationTime(stn, 'idle');
                        this._tryStartProcessing(stn.id);
                        return;
                    }
                } else {
                    this.escapedDefects++;
                    job._hasDefect = true;
                }
            }
        }

        // Check: did this station catch a defect from UPSTREAM?
        if (job._hasDefect && Math.random() < stn.defectDetectionRate * 0.5) {
            // Caught an escaped defect! Rework.
            this.detectedDownstream++;
            job._hasDefect = false;
            stn.stats.reworked++;
            stn.stats.processed++;
            stn.currentJob = null;
            if (stn.reworkTarget && this.stations.has(stn.reworkTarget)) {
                this._deliverJob(job, stn.reworkTarget);
            } else {
                this._enqueueJob(stn, job);
            }
            this._updateStationTime(stn, 'idle');
            this._tryStartProcessing(stn.id);
            return;
        }

        // Inline SPC: track quality measurement and check Western Electric rules
        if (stn.spcEnabled) {
            // Simulated measurement: centered at 0, shifted by drift + fatigue + material
            const processNoise = (Math.random() + Math.random() + Math.random() - 1.5) * 2; // approx normal
            const driftShift = stn.driftAccumulated * 10; // drift shifts the mean
            const fatigueShift = fatigueErrorBoost * 2;
            const measurement = processNoise + driftShift + fatigueShift;

            // Measurement lag: buffer measurements with delay before they appear on SPC chart
            if (stn.measurementDelay > 0) {
                stn._pendingMeasurements.push({
                    value: measurement,
                    availableAt: this.clock + stn.measurementDelay,
                    driftAtSample: stn.driftAccumulated,
                    badMaterial: !!job._badMaterial,
                });
                // Parts produced while waiting for data — the containment scope
                stn.containmentScope++;
            } else {
                stn.spcMeasurements.push(measurement);
            }

            // Flush matured pending measurements into SPC chart
            while (stn._pendingMeasurements.length > 0 && stn._pendingMeasurements[0].availableAt <= this.clock) {
                const m = stn._pendingMeasurements.shift();
                stn.spcMeasurements.push(m.value);
            }
            if (stn.spcMeasurements.length > 25) {
                stn.spcMeasurements.splice(0, stn.spcMeasurements.length - 25); // rolling window
            }

            // Western Electric rules (using 3σ = 3.0 for control limits)
            const signal = this._checkSPCRules(stn.spcMeasurements);
            if (signal) {
                stn.spcSignalCount++;
                // Is this a real problem or false alarm?
                // Real if drift is above threshold OR bad material
                const isReal = stn.driftAccumulated > 0.02 || job._badMaterial;
                if (!isReal) stn.spcFalseAlarms++;

                // Log containment scope — how many parts were produced before detection
                if (stn.measurementDelay > 0) {
                    this.simLog.push({
                        time: this.clock,
                        msg: `SPC signal at ${stn.name}: ${stn.containmentScope} parts in containment scope (${stn.measurementDelay}s measurement lag)`,
                    });
                    stn.containmentScope = 0; // reset after detection
                }

                // Stop machine for investigation
                stn.stats.processed++;
                stn.currentJob = null;
                this._updateStationTime(stn, 'down');
                // If real, reset drift (calibration triggered by SPC)
                if (isReal) {
                    stn.driftAccumulated = 0;
                    stn.jobsSinceCalibration = 0;
                }
                // Schedule resume after investigation
                this.eventQueue.push({
                    time: this.clock + stn.spcInvestigationTime,
                    type: 'repair', stationId: stn.id,
                });
                // Route the current job downstream still (it passed)
                const ds = this._getDownstream(stn.id);
                if (ds.length > 0) {
                    const t = this._pickTarget(ds, job);
                    this._deliverJob(job, t.toId);
                } else {
                    job.completedAt = this.clock;
                    this.completedJobs.push(job);
                }
                return;
            }
        }

        const downstream = this._getDownstream(stn.id);

        if (downstream.length === 0) {
            // No downstream — job completed
            if (job._hasDefect || job._contaminationDefect) this.customerReturns++;
            job.completedAt = this.clock;
            if (job.dueDate != null) {
                if (this.clock <= job.dueDate) this.onTimeCount++;
                else this.lateOrderCount++;
            }
            this._recordCustomerDelivery(job);
            this.completedJobs.push(job);
            stn.currentJob = null;
            stn.stats.processed++;
            this._updateStationTime(stn, 'idle');
            this._tryStartProcessing(stn.id);
            return;
        }

        // Try to move job downstream
        const target = this._pickTarget(downstream, job);
        const targetStn = this.stations.get(target.toId);
        const transportTime = this._getTransportTime(stn.id, target.toId);

        if (targetStn && targetStn.type === 'sink') {
            if (job._hasDefect || job._contaminationDefect) this.customerReturns++;
            this._recordCustomerDelivery(job);
            stn.currentJob = null;
            stn.stats.processed++;
            this._updateStationTime(stn, 'idle');
            this._tryStartProcessing(stn.id);
            if (transportTime > 0) {
                this._requestTransport(job, targetStn.id, transportTime);
            } else {
                job.completedAt = this.clock;
                job.stationId = targetStn.id;
                this.completedJobs.push(job);
            }
            return;
        }

        if (targetStn && this._canAccept(targetStn.id)) {
            stn.currentJob = null;
            stn.stats.processed++;
            this._updateStationTime(stn, 'idle');
            this._tryStartProcessing(stn.id);
            if (transportTime > 0) {
                this._requestTransport(job, targetStn.id, transportTime);
            } else {
                targetStn.queue.push(job);
                job.stationId = targetStn.id;
                this._tryStartProcessing(targetStn.id);
            }
        } else {
            // Blocked — hold job until downstream opens
            stn.blockedJob = job;
            this._updateStationTime(stn, 'blocked');
        }
    }

    _handleBreakdown(event) {
        const stn = this.stations.get(event.stationId);
        if (!stn) return;
        if (stn.state === 'down') return; // already down (e.g., PM in progress)

        stn.unplannedDownCount++;
        // If currently processing, job goes back to queue
        if (stn.currentJob) {
            this._releaseOperator(stn.id);
            this._enqueueJob(stn, stn.currentJob);
            stn.currentJob = null;
        }
        this._updateStationTime(stn, 'down');
        const repairTime = stn.mttr ? this._sampleExponential(stn.mttr) : 60;

        // Shared maintenance crew: if finite, request a technician
        if (this.maintenanceCrew.size > 0) {
            this._requestMaintenance(stn.id, repairTime);
        } else {
            // Unlimited crew — immediate repair
            this.eventQueue.push({ time: this.clock + repairTime, type: 'repair', stationId: stn.id });
        }

        // Dynamic rerouting: move queued jobs to sibling machines
        this._rerouteOnFailure(stn);
    }

    _handleRepair(event) {
        const stn = this.stations.get(event.stationId);
        if (!stn) return;

        // Reset operating time after repair (machine is "like new" for Weibull purposes)
        stn.operatingTime = 0;
        this._updateStationTime(stn, 'idle');
        // Schedule next breakdown using Weibull (operating-time based)
        if (stn.mtbf && stn.mtbf > 0) {
            const nextBreakdown = this._sampleWeibull(stn.mtbf, stn.weibullBeta || 1);
            this.eventQueue.push({
                time: this.clock + nextBreakdown,
                type: 'breakdown', stationId: stn.id,
            });
        }
        this._tryStartProcessing(stn.id);

        // Release maintenance technician back to pool
        if (this.maintenanceCrew.size > 0) {
            this.maintenanceCrew.available++;
            this.maintenanceCrew.repairsCompleted++;
            this._dispatchNextMaintenance();
        }
    }

    _handlePMStart(event) {
        const stn = this.stations.get(event.stationId);
        if (!stn) return;
        if (stn.state === 'down') return; // can't PM if already broken

        stn.pmCount++;
        // If currently processing, job goes back to queue
        if (stn.currentJob) {
            this._releaseOperator(stn.id);
            this._enqueueJob(stn, stn.currentJob);
            stn.currentJob = null;
        }
        this._updateStationTime(stn, 'down'); // PM counts as down visually, tracked separately
        stn.stats.plannedDown += stn.pmDuration;

        // PMs also require a maintenance technician
        if (this.maintenanceCrew.size > 0) {
            this._requestMaintenance(stn.id, stn.pmDuration, true);
        } else {
            this.eventQueue.push({ time: this.clock + stn.pmDuration, type: 'pm_end', stationId: stn.id });
        }
    }

    _handlePMEnd(event) {
        const stn = this.stations.get(event.stationId);
        if (!stn) return;

        // PM resets degradation — machine is restored
        stn.operatingTime = 0;
        this._updateStationTime(stn, 'idle');
        // Schedule next PM
        if (stn.pmInterval > 0) {
            this.eventQueue.push({ time: this.clock + stn.pmInterval, type: 'pm_start', stationId: stn.id });
        }
        this._tryStartProcessing(stn.id);
        this._unblockUpstream(stn.id);

        // Release maintenance technician
        if (this.maintenanceCrew.size > 0) {
            this.maintenanceCrew.available++;
            this.maintenanceCrew.repairsCompleted++;
            this._dispatchNextMaintenance();
        }
    }

    // Shared maintenance crew: request a technician for breakdown or PM
    _requestMaintenance(stationId, repairTime, isPM = false) {
        const mc = this.maintenanceCrew;
        const eventType = isPM ? 'pm_end' : 'repair';

        if (mc.available > 0) {
            // Technician available — start repair immediately
            mc.available--;
            this.eventQueue.push({ time: this.clock + repairTime, type: eventType, stationId });
        } else {
            // All technicians busy — machine waits in repair queue
            mc.repairQueue.push({
                stationId,
                requestedAt: this.clock,
                repairTime,
                isPM,
            });
            // Sort queue: unplanned breakdowns before PMs (breakdowns are more urgent)
            mc.repairQueue.sort((a, b) => {
                if (a.isPM !== b.isPM) return a.isPM ? 1 : -1;
                return a.requestedAt - b.requestedAt; // FIFO within type
            });
        }
    }

    // Dispatch the next machine in the repair queue when a tech becomes available
    _dispatchNextMaintenance() {
        const mc = this.maintenanceCrew;
        if (mc.available <= 0 || mc.repairQueue.length === 0) return;

        const next = mc.repairQueue.shift();
        mc.available--;
        mc.totalWaitTime += (this.clock - next.requestedAt);
        const eventType = next.isPM ? 'pm_end' : 'repair';
        this.eventQueue.push({ time: this.clock + next.repairTime, type: eventType, stationId: next.stationId });
    }

    // Release shared tool after processing and dispatch next queued machine
    _releaseSharedTool(stn) {
        if (!stn._holdingTool) return;
        const tool = this.sharedTools.find(t => t.id === stn._holdingTool);
        if (!tool) { stn._holdingTool = null; return; }
        tool.available++;
        tool.useCount++;
        stn._holdingTool = null;
        // Dispatch next waiting machine
        while (tool.toolQueue.length > 0 && tool.available > 0) {
            const next = tool.toolQueue.shift();
            tool.totalWaitTime += this.clock - next.requestedAt;
            // Trigger that machine to try processing again
            this._tryStartProcessing(next.stationId);
            // If that machine actually acquired the tool, available will have decremented
            if (tool.available <= 0) break;
        }
    }

    // Dynamic rerouting: when a machine fails, move its queued jobs to sibling machines
    // (siblings = other machines that share the same upstream node)
    _rerouteOnFailure(failedStn) {
        if (failedStn.queue.length === 0) return;

        // Find upstream nodes of this station
        const upstreams = this.reverseAdj.get(failedStn.id) || [];
        if (upstreams.length === 0) return;

        // Find sibling machines: other downstream targets of the same upstream(s)
        const siblings = [];
        for (const upId of upstreams) {
            const downstream = this._getDownstream(upId);
            for (const d of downstream) {
                if (d.toId === failedStn.id) continue;
                const stn = this.stations.get(d.toId);
                if (stn && stn.type === 'machine' && stn.state !== 'down') {
                    siblings.push(stn);
                }
            }
        }
        if (siblings.length === 0) return;

        // Reroute jobs to siblings with shortest queues
        const rerouted = [];
        while (failedStn.queue.length > 0 && siblings.length > 0) {
            // Pick sibling with shortest queue
            let best = siblings[0];
            for (const s of siblings) {
                if (s.queue.length < best.queue.length) best = s;
            }
            // Only reroute if sibling queue isn't already longer than the failed machine's
            if (best.queue.length >= failedStn.queue.length) break;

            const job = failedStn.queue.shift();
            this._enqueueJob(best, job);
            job.stationId = best.id;
            rerouted.push(job.id);
            this._tryStartProcessing(best.id);
        }
    }

    // Utility system failure: takes down ALL machines connected to that utility
    _handleUtilityFailure(event) {
        const util = this.utilitySystems.find(u => u.id === event.utilityId);
        if (!util || util.state === 'down') return;

        util.state = 'down';
        util.failureCount++;
        util.downSince = this.clock;

        // Take down every affected machine
        for (const stnId of util.affectedMachines) {
            const stn = this.stations.get(stnId);
            if (!stn || stn.type !== 'machine' || stn.state === 'down') continue;

            // If processing, interrupt — job goes back to queue
            if (stn.currentJob) {
                this._releaseOperator(stn.id);
                this._enqueueJob(stn, stn.currentJob);
                stn.currentJob = null;
            }
            this._updateStationTime(stn, 'down');
            // Mark as utility-caused so regular repair doesn't trigger
            stn._utilityDown = true;
        }

        // Schedule utility restore
        const restoreTime = this._sampleExponential(util.mttr);
        this.eventQueue.push({
            time: this.clock + restoreTime,
            type: 'utility_restore',
            utilityId: util.id,
        });
    }

    _handleUtilityRestore(event) {
        const util = this.utilitySystems.find(u => u.id === event.utilityId);
        if (!util || util.state !== 'down') return;

        util.state = 'up';
        util.totalDowntime += (this.clock - util.downSince);
        util.downSince = null;

        // Restore all affected machines
        for (const stnId of util.affectedMachines) {
            const stn = this.stations.get(stnId);
            if (!stn || stn.type !== 'machine') continue;
            if (!stn._utilityDown) continue; // wasn't downed by this utility

            stn._utilityDown = false;
            this._updateStationTime(stn, 'idle');
            this._tryStartProcessing(stn.id);
            this._unblockUpstream(stn.id);
        }

        // Schedule next failure
        if (util.mtbf > 0) {
            const nextFailure = this._sampleExponential(util.mtbf);
            this.eventQueue.push({
                time: this.clock + nextFailure,
                type: 'utility_failure',
                utilityId: util.id,
            });
        }
    }

    // Inspector pool: request an inspection for a completed job
    _requestInspection(job, stationId) {
        const pool = this.inspectorPool;
        const inspectionTime = 15 + Math.random() * 15; // 15-30s per inspection
        if (pool.available > 0) {
            pool.available--;
            this.eventQueue.push({
                time: this.clock + inspectionTime,
                type: 'inspection_complete',
                job, stationId, inspectionTime,
            });
        } else {
            pool.inspectionQueue.push({
                job, stationId, requestedAt: this.clock, inspectionTime,
            });
        }
    }

    _handleInspectionComplete(event) {
        const pool = this.inspectorPool;
        const job = event.job;
        const stn = this.stations.get(event.stationId);
        pool.available++;
        pool.inspectionsCompleted++;

        // Now do the quality check
        const scrapRate = job._effectiveScrapRate || 0;
        const defectRate = job._effectiveDefectRate || 0;
        const reworkRate = job._effectiveReworkRate || 0;
        const detectionRate = job._defectDetectionRate ?? 1.0;

        if (defectRate > 0) {
            // CR-3: 3-stage quality model
            if (Math.random() < scrapRate) {
                if (stn) stn.stats.scrapped++;
                this.costs.scrapWaste += (job._materialCost || 0);
                this._dispatchNextInspection();
                return;
            }
            if (Math.random() < defectRate) {
                const detected = Math.random() < detectionRate;
                if (detected) {
                    if (reworkRate > 0 && Math.random() < reworkRate) {
                        if (stn) {
                            stn.stats.reworked++;
                            if (stn.reworkTarget && this.stations.has(stn.reworkTarget)) {
                                this._deliverJob(job, stn.reworkTarget);
                            } else {
                                this._enqueueJob(stn, job);
                                this._tryStartProcessing(stn.id);
                            }
                        }
                    } else {
                        if (stn) stn.stats.scrapped++;
                        this.costs.scrapWaste += (job._materialCost || 0);
                    }
                    this._dispatchNextInspection();
                    return;
                }
                job._hasDefect = true;
                this.escapedDefects++;
            }
        } else {
            // Legacy: single roll for scrap + rework
            const roll = Math.random();
            const isDefective = roll < scrapRate + reworkRate;
            const isScrap = roll < scrapRate;

            if (isDefective) {
                const detected = Math.random() < detectionRate;
                if (detected) {
                    if (isScrap) {
                        if (stn) stn.stats.scrapped++;
                        this.costs.scrapWaste += (job._materialCost || 0);
                    } else {
                        if (stn) {
                            stn.stats.reworked++;
                            if (stn.reworkTarget && this.stations.has(stn.reworkTarget)) {
                                this._deliverJob(job, stn.reworkTarget);
                            } else {
                                this._enqueueJob(stn, job);
                                this._tryStartProcessing(stn.id);
                            }
                        }
                    }
                    this._dispatchNextInspection();
                    return;
                }
                job._hasDefect = true;
                this.escapedDefects++;
            }
        }

        // Route downstream (passed or escaped)
        if (stn) {
            const downstream = this._getDownstream(stn.id);
            if (downstream.length > 0) {
                const target = this._pickTarget(downstream, job);
                const targetStn = this.stations.get(target.toId);
                const transportTime = this._getTransportTime(stn.id, target.toId);
                if (targetStn && targetStn.type === 'sink') {
                    if (job._hasDefect || job._contaminationDefect) this.customerReturns++;
                    this._recordCustomerDelivery(job);
                    job.completedAt = this.clock;
                    this.completedJobs.push(job);
                } else if (transportTime > 0) {
                    this._requestTransport(job, target.toId, transportTime);
                } else {
                    this._deliverJob(job, target.toId);
                }
            } else {
                if (job._hasDefect || job._contaminationDefect) this.customerReturns++;
                this._recordCustomerDelivery(job);
                job.completedAt = this.clock;
                if (job.dueDate != null) {
                    if (this.clock <= job.dueDate) this.onTimeCount++;
                    else this.lateOrderCount++;
                }
                this.completedJobs.push(job);
            }
        }
        this._dispatchNextInspection();
    }

    _dispatchNextInspection() {
        const pool = this.inspectorPool;
        if (pool.inspectionQueue.length === 0 || pool.available <= 0) return;
        const next = pool.inspectionQueue.shift();
        pool.totalWaitTime += this.clock - next.requestedAt;
        pool.available--;
        this.eventQueue.push({
            time: this.clock + next.inspectionTime,
            type: 'inspection_complete',
            job: next.job,
            stationId: next.stationId,
            inspectionTime: next.inspectionTime,
        });
    }

    // ===== MISSION EVENT HANDLING =====
    _handleMissionEvent(event) {
        const te = event.timelineEntry;
        if (!te || te.fired) return;
        te.fired = true;

        const alert = {
            id: ++this.alertIdCounter,
            time: this.clock,
            severity: te.severity || 'warning',
            message: te.message || 'Mission event occurred',
            type: te.type,
            target: te.target,
            acknowledged: false,
        };
        this.missionAlerts.push(alert);

        if (te.effect) {
            this._applyTimelineEffect(te.effect, te.target);
        }

        if (this.onAlert) this.onAlert(alert);

        if (te.severity === 'critical' && this.onPause) {
            this.paused = true;
            this.onPause();
        }
    }

    _applyTimelineEffect(effect, target) {
        const stn = target ? this.stations.get(target) : null;

        switch (effect.type) {
            case 'breakdown':
                if (stn && stn.type === 'machine' && stn.state !== 'down') {
                    stn.unplannedDownCount = (stn.unplannedDownCount || 0) + 1;
                    if (stn.currentJob) {
                        this._releaseOperator(stn.id);
                        this._enqueueJob(stn, stn.currentJob);
                        stn.currentJob = null;
                    }
                    this._updateStationTime(stn, 'down');
                    if (!effect.noAutoRepair) {
                        const repairTime = effect.repairTime || (stn.mttr ? this._sampleExponential(stn.mttr) : 300);
                        if (this.maintenanceCrew.size > 0) {
                            this._requestMaintenance(stn.id, repairTime);
                        } else {
                            this.eventQueue.push({ time: this.clock + repairTime, type: 'repair', stationId: stn.id });
                        }
                    }
                }
                break;

            case 'param_change':
                if (stn && effect.param && effect.value !== undefined) {
                    stn[effect.param] = effect.value;
                }
                break;

            case 'operator_quit': {
                let quit = 0;
                const count = effect.count || 1;
                for (const op of this.operators) {
                    if (op.status !== 'quit' && quit < count) {
                        if (op.assignedTo) {
                            const assignedStn = this.stations.get(op.assignedTo);
                            if (assignedStn && assignedStn.currentJob) {
                                this._enqueueJob(assignedStn, assignedStn.currentJob);
                                assignedStn.currentJob = null;
                                this._updateStationTime(assignedStn, 'starved');
                            }
                        }
                        op.status = 'quit';
                        op.assignedTo = null;
                        quit++;
                    }
                }
                break;
            }

            case 'demand_spike':
                if (stn && stn.type === 'source' && effect.multiplier) {
                    const oldRate = stn.arrivalRate;
                    stn.arrivalRate = stn.arrivalRate / effect.multiplier;
                    if (effect.duration) {
                        this.eventQueue.push({
                            time: this.clock + effect.duration,
                            type: 'mission_event',
                            timelineEntry: {
                                at: 0, severity: 'info',
                                message: 'Demand spike subsiding — returning to normal rate',
                                effect: { type: 'param_change', param: 'arrivalRate', value: oldRate },
                                fired: false,
                            },
                        });
                    }
                }
                break;

            case 'utility_failure': {
                const util = this.utilitySystems.find(u => u.id === effect.utilityId);
                if (util && util.state === 'up') {
                    util.state = 'down';
                    util.failureCount++;
                    util.downSince = this.clock;
                    for (const machId of util.affectedMachines) {
                        const m = this.stations.get(machId);
                        if (m && m.type === 'machine') {
                            m._utilityDown = true;
                            if (m.currentJob) {
                                this._releaseOperator(m.id);
                                this._enqueueJob(m, m.currentJob);
                                m.currentJob = null;
                            }
                            this._updateStationTime(m, 'down');
                        }
                    }
                    const repairTime = effect.repairTime || (util.mttr ? this._sampleExponential(util.mttr) : 600);
                    this.eventQueue.push({ time: this.clock + repairTime, type: 'utility_repair', utilityId: util.id });
                }
                break;
            }

            case 'quality_excursion':
                if (stn && stn.type === 'machine') {
                    const originalScrap = stn.scrapRate;
                    stn.scrapRate = effect.scrapRate || 0.3;
                    if (effect.duration) {
                        this.eventQueue.push({
                            time: this.clock + effect.duration,
                            type: 'mission_event',
                            timelineEntry: {
                                at: 0, severity: 'info',
                                message: `Quality excursion resolved on ${stn.name}`,
                                effect: { type: 'param_change', param: 'scrapRate', value: originalScrap },
                                fired: false,
                            },
                        });
                    }
                }
                break;

            case 'supplier_disruption':
                if (stn && stn.type === 'source') {
                    stn._paused = true;
                    if (effect.duration) {
                        this.eventQueue.push({
                            time: this.clock + effect.duration,
                            type: 'mission_event',
                            timelineEntry: {
                                at: 0, severity: 'info',
                                message: 'Supplier deliveries resumed',
                                effect: { type: 'param_change', param: '_paused', value: false },
                                fired: false,
                            },
                        });
                    }
                }
                break;
        }
    }

    // Engineering Change Order: obsolete WIP in buffers
    _handleECO() {
        if (this.ecoConfig.rate <= 0) return;

        // Pick a random product type to change
        const productTypes = new Set();
        for (const [, stn] of this.stations) {
            if (stn.type === 'source' && stn.productTypes) {
                for (const pt of stn.productTypes) productTypes.add(pt.name);
            }
        }
        const types = [...productTypes];
        if (types.length === 0) return;
        const affectedType = types[Math.floor(Math.random() * types.length)];

        // Scrap all queued jobs of this product type across all machines
        let scrapped = 0;
        for (const [, stn] of this.stations) {
            if (stn.type !== 'machine') continue;
            const before = stn.queue.length;
            stn.queue = stn.queue.filter(j => {
                if (j.productType === affectedType) {
                    scrapped++;
                    stn.stats.scrapped++;
                    this.costs.scrapWaste += (j._materialCost || 0);
                    return false;
                }
                return true;
            });
        }

        this.ecoCount++;
        this.ecoScrappedWIP += scrapped;
        this.simLog.push({
            time: this.clock,
            msg: `ECO issued for "${affectedType}": ${scrapped} WIP parts obsoleted and scrapped`,
        });

        // Schedule next ECO
        const nextEco = this._sampleExponential(3600 / this.ecoConfig.rate);
        this.eventQueue.push({ time: this.clock + nextEco, type: 'eco_event' });
    }

    // Management policy oscillation: reactive decisions that create instability
    _handleManagementReview() {
        const mgmt = this.management;
        if (mgmt.reactivity === 'off') return;

        const state = this.getState();
        const wip = state.wip;
        const yield_ = state.yieldRate;
        const onTime = state.onTimeDelivery;
        const sensitivity = mgmt.reactivity === 'panicking' ? 0.6 : mgmt.reactivity === 'nervous' ? 0.8 : 1.0;

        // Reaction 1: WIP is rising → slash batch sizes (causes more changeovers)
        if (wip > mgmt.lastWIP * 1.2 * sensitivity && wip > 10) {
            for (const [id, stn] of this.stations) {
                if (stn.type === 'machine' && stn.batchSize > 1) {
                    stn.batchSize = Math.max(1, Math.floor(stn.batchSize * 0.7));
                }
            }
            mgmt.policyLog.push({ time: this.clock, action: 'batch_slash', detail: `WIP ${wip.toFixed(0)} rising — slashed batch sizes` });
            mgmt.interventionCount++;
        }

        // Reaction 2: Yield dropping → tighten SPC (more investigation stops)
        if (yield_ < mgmt.lastYield * sensitivity && yield_ < 0.95) {
            for (const [id, stn] of this.stations) {
                if (stn.type === 'machine' && !stn.spcEnabled && stn.scrapRate > 0) {
                    stn.spcEnabled = true;
                    stn.spcInvestigationTime = 180; // aggressive 3min investigations
                }
            }
            mgmt.policyLog.push({ time: this.clock, action: 'tighten_spc', detail: `Yield ${(yield_ * 100).toFixed(1)}% — enabled SPC on all machines` });
            mgmt.interventionCount++;
        }

        // Reaction 3: On-time delivery tanking → authorize rush on everything (expediting spiral)
        if (onTime < 0.8 * sensitivity) {
            let promoted = 0;
            for (const [id, stn] of this.stations) {
                if (stn.type !== 'machine') continue;
                for (const job of stn.queue) {
                    if (job.priority === 'normal') {
                        job.priority = 'rush';
                        job.productColor = '#ff6b6b';
                        promoted++;
                        this.promotedToRush++;
                    }
                }
            }
            if (promoted > 0) {
                mgmt.policyLog.push({ time: this.clock, action: 'rush_everything', detail: `OTD ${(onTime * 100).toFixed(0)}% — promoted ${promoted} jobs to rush` });
                mgmt.interventionCount++;
            }
        }

        // Reaction 4: WIP falling + machines idle → increase batch sizes (whiplash from reaction 1)
        if (wip < mgmt.lastWIP * 0.7 * sensitivity && wip < 5) {
            for (const [id, stn] of this.stations) {
                if (stn.type === 'machine') {
                    stn.batchSize = Math.min(20, Math.max(stn.batchSize, Math.ceil(stn.batchSize * 1.5)));
                }
            }
            mgmt.policyLog.push({ time: this.clock, action: 'batch_increase', detail: `WIP ${wip.toFixed(0)} low — increased batch sizes` });
            mgmt.interventionCount++;
        }

        // Update last-seen metrics
        mgmt.lastWIP = wip;
        mgmt.lastYield = yield_;
        mgmt.lastOnTime = onTime;

        // Schedule next review
        this.eventQueue.push({ time: this.clock + mgmt.reviewInterval, type: 'management_review' });
    }

    _handleBreakStart(event) {
        const stn = this.stations.get(event.stationId);
        if (!stn || stn.type !== 'machine') return;

        // If currently processing, the job pauses — it'll resume after break
        // We don't cancel the end_processing event; instead, break_end will re-check
        this._updateStationTime(stn, 'onBreak');
    }

    _handleBreakEnd(event) {
        const stn = this.stations.get(event.stationId);
        if (!stn || stn.type !== 'machine') return;
        if (stn.state !== 'onBreak') return; // might have been overridden

        this._updateStationTime(stn, 'idle');

        // SHIFT HANDOVER LOSS: new crew may not know what was last running
        if (stn.handoverLossRate > 0 && Math.random() < stn.handoverLossRate) {
            // Crew forgot the last product type — will trigger unnecessary changeover
            stn.lastProductType = null;
            stn.handoverLossCount++;
        }

        // Reset first-article counter (machine was idle during break)
        stn.partsAfterRestart = 0;

        // Machine warmup after idle
        if (stn.warmupTime > 0) {
            this._updateStationTime(stn, 'setup'); // warmup counts as setup time
            this.eventQueue.push({
                time: this.clock + stn.warmupTime,
                type: 'changeover_end',  // reuse changeover_end to resume after warmup
                stationId: stn.id,
            });
            return;
        }

        this._tryStartProcessing(stn.id);
        this._unblockUpstream(stn.id);
    }

    _handleWorkforceCheck() {
        // Overtime authorization check: if WIP exceeds threshold, authorize OT
        const prevOT = this.overtimeActive;
        if (this.overtimeConfig.wipThreshold > 0) {
            const currentWIP = this._currentWIP();
            this.overtimeActive = currentWIP >= this.overtimeConfig.wipThreshold;
            if (this.overtimeActive && !prevOT) {
                this.overtimeShifts++;
                this.workforceLog.push({
                    time: this.clock, event: 'overtime',
                    detail: `OT authorized — WIP ${currentWIP} ≥ threshold ${this.overtimeConfig.wipThreshold}`,
                });
            }
        }

        // Update morale for all active operators before attendance checks
        const activeOps = this.operators.filter(o => o.status !== 'quit');
        const quitCount = this.operators.filter(o => o.status === 'quit').length;
        const totalOriginal = this.operators.length;
        const quitRatio = totalOriginal > 0 ? quitCount / totalOriginal : 0;

        for (const op of activeOps) {
            // Morale impacts: peer quits, overtime, breakdowns at their station
            let moraleDelta = 0;
            moraleDelta -= quitRatio * 0.15;               // peer quits drag morale down
            moraleDelta -= op.overtimeHours * 0.02;         // overtime burns people out
            moraleDelta += 0.02;                             // small natural recovery per shift
            // OT morale hit: mandatory OT hurts morale significantly
            if (this.overtimeActive) moraleDelta -= 0.08;
            // Breakdowns at assigned station frustrate operators
            if (op.assignedTo) {
                const stn = this.stations.get(op.assignedTo);
                if (stn && stn.unplannedDownCount > 0) moraleDelta -= 0.01 * stn.unplannedDownCount;
            }
            op.morale = Math.max(0.1, Math.min(1, op.morale + moraleDelta));

            // Skill forgetting: skills decay when not practiced
            // Exponential decay: lose ~5% of skill per shift (8hr) of non-practice
            // Floor at 0.05 (muscle memory never fully disappears)
            for (const stnId of Object.keys(op.skills)) {
                const lastPractice = op.lastWorkedAt[stnId] || 0;
                const hoursSince = (this.clock - lastPractice) / 3600;
                if (hoursSince > 8 && op.skills[stnId] > 0.05) {
                    const decayRate = 0.005 * (hoursSince / 8); // accelerates with disuse
                    op.skills[stnId] = Math.max(0.05, op.skills[stnId] - decayRate);
                }
            }

            // Reset shift fatigue (new shift) — unless OT is active
            if (this.overtimeActive) {
                // OT: partial fatigue reset (they're tired from the extra hours)
                op.fatigue = Math.min(1, op.fatigue * 0.6 + 0.15);  // carry over fatigue
                op.overtimeHours += this.overtimeConfig.maxOTHours;
                this.totalOTHours += this.overtimeConfig.maxOTHours;
            } else {
                op.fatigue = 0;
            }
            op.shiftStart = this.clock;
            op.jobsThisShift = 0;
        }

        // Shift change: check for call-offs, early leaves, quits
        for (const op of this.operators) {
            if (op.status === 'quit') continue;

            // Quit rate amplified by low morale: base rate × (2 - morale)
            // Mandatory OT further amplifies quit rate (1.5× during OT)
            const otQuitMultiplier = this.overtimeActive ? 1.5 : 1;
            const effectiveQuitRate = this.quitRate * (2 - op.morale) * otQuitMultiplier;
            if (Math.random() < effectiveQuitRate) {
                const wasWorking = op.assignedTo;
                op.status = 'quit';
                op.assignedTo = null;
                // TRIBAL KNOWLEDGE LOSS: document what knowledge was lost
                const lostTweaks = Object.entries(op.processTweaks || {})
                    .filter(([_, v]) => v > 0.1)
                    .map(([stnId, v]) => `${this.stations.get(stnId)?.name || stnId}: ${(v * 100).toFixed(0)}%`)
                    .join(', ');
                this.workforceLog.push({
                    time: this.clock, event: 'quit', operator: op.name,
                    detail: `${op.name} quit permanently${lostTweaks ? '. KNOWLEDGE LOST: ' + lostTweaks : ''}`,
                });
                // If they were mid-job, the machine is now stuck
                if (wasWorking) {
                    const stn = this.stations.get(wasWorking);
                    if (stn && stn.state === 'processing') {
                        if (stn.currentJob) {
                            this._enqueueJob(stn, stn.currentJob);
                            stn.currentJob = null;
                        }
                        this._updateStationTime(stn, 'starved');
                        this._tryStartProcessing(stn.id);
                    }
                }
                continue;
            }

            // Call-off rate amplified by low morale
            const effectiveCalloff = this.calloffRate * (2 - op.morale);
            if (op.status === 'available' && Math.random() < effectiveCalloff) {
                op.status = 'absent';
                this.workforceLog.push({
                    time: this.clock, event: 'calloff', operator: op.name,
                    detail: `${op.name} called off`,
                });
                // Schedule return next shift (8hr)
                this.eventQueue.push({
                    time: this.clock + 28800,
                    type: 'operator_return',
                    operatorId: op.id,
                });
                continue;
            }

            // Early leave: random chance a working operator leaves mid-shift
            if (op.status === 'available' || op.status === 'busy') {
                if (Math.random() < this.calloffRate * 0.3) {
                    const leaveIn = 3600 + Math.random() * 14400; // 1-5 hrs into shift
                    this.eventQueue.push({
                        time: this.clock + leaveIn,
                        type: 'early_leave',
                        operatorId: op.id,
                    });
                }
            }

            // Return absent operators
            if (op.status === 'absent') {
                // They might come back this shift check
                op.status = 'available';
                this.workforceLog.push({
                    time: this.clock, event: 'return', operator: op.name,
                    detail: `${op.name} returned`,
                });
            }
        }

        // With operators back, try to start starving machines
        for (const [id, stn] of this.stations) {
            if (stn.type === 'machine' && (stn.state === 'starved' || stn.state === 'idle') && stn.queue.length > 0) {
                this._tryStartProcessing(id);
            }
        }
    }

    _handleEarlyLeave(event) {
        const op = this.operators.find(o => o.id === event.operatorId);
        if (!op || op.status === 'quit' || op.status === 'absent') return;

        const wasWorking = op.assignedTo;
        op.status = 'absent';
        op.assignedTo = null;
        this.workforceLog.push({
            time: this.clock, event: 'early_leave', operator: op.name,
            detail: `${op.name} left early`,
        });

        // Schedule return next shift
        this.eventQueue.push({
            time: this.clock + 28800,
            type: 'operator_return',
            operatorId: op.id,
        });

        // If they were mid-job, machine needs another operator
        if (wasWorking) {
            const stn = this.stations.get(wasWorking);
            if (stn && stn.state === 'processing' && stn.currentJob) {
                this._enqueueJob(stn, stn.currentJob);
                stn.currentJob = null;
                this._updateStationTime(stn, 'starved');
                this._tryStartProcessing(stn.id);
            }
        }
    }

    _handleOperatorReturn(event) {
        const op = this.operators.find(o => o.id === event.operatorId);
        if (!op || op.status === 'quit') return;
        if (op.status === 'absent') {
            op.status = 'available';
            this.workforceLog.push({
                time: this.clock, event: 'return', operator: op.name,
                detail: `${op.name} returned`,
            });
            // Try starving machines
            for (const [id, stn] of this.stations) {
                if (stn.type === 'machine' && stn.state === 'starved' && stn.queue.length > 0) {
                    this._tryStartProcessing(id);
                }
            }
        }
    }

    _handleTrainingComplete(event) {
        const op = this.operators.find(o => o.id === event.operatorId);
        if (!op || op.status === 'quit') return;

        const prevSkill = op.skills[event.stationId] ?? 0;
        op.skills[event.stationId] = Math.min(1, prevSkill + 0.3); // Training bump
        op.status = 'available';
        op.trainingTarget = null;
        this.workforceLog.push({
            time: this.clock, event: 'training_complete', operator: op.name,
            detail: `${op.name} trained on ${this.stations.get(event.stationId)?.name || event.stationId}: ${(prevSkill * 100).toFixed(0)}% → ${(op.skills[event.stationId] * 100).toFixed(0)}%`,
        });

        // Now available — try machines
        for (const [id, stn] of this.stations) {
            if (stn.type === 'machine' && stn.state === 'starved' && stn.queue.length > 0) {
                this._tryStartProcessing(id);
            }
        }
    }

    _handleRecordStats() {
        if (this.clock < this.warmupTime) {
            // Reset stats at warmup boundary
            if (this.clock + 10 >= this.warmupTime) {
                for (const [, stn] of this.stations) {
                    stn.stats = this._emptyStats();
                    stn.stateStartTime = this.warmupTime;
                    // Reset degradation/drift tracking
                    if (stn.type === 'machine') {
                        stn.driftAccumulated = 0;
                        stn.jobsSinceCalibration = 0;
                        stn.pmCount = 0;
                        stn.unplannedDownCount = 0;
                    }
                    // Reset FG tracking (keep current inventory as starting point)
                    if (stn.type === 'sink' && stn.sinkMode === 'fg_warehouse') {
                        stn.stockouts = {};
                        stn.demandFilled = {};
                        stn.fgHistory = [];
                    }
                }
                this.completedJobs = [];
                this.escapedDefects = 0;
                this.detectedDownstream = 0;
                this.customerReturns = 0;
                this.rushOrderCount = 0;
                this.lateOrderCount = 0;
                this.onTimeCount = 0;
                this.promotedToRush = 0;
                this.expiredWIP = 0;
                this.ecoCount = 0;
                this.ecoScrappedWIP = 0;
                this.customerSatisfaction = 1.0;
                this.totalRevenue = 0;
                this.totalReturnCost = 0;
                this.customerLostOrders = 0;
                this.costs = { labor: 0, material: 0, scrapWaste: 0, holdingCost: 0, overtimeCost: 0, totalCost: 0 };
                this.history = { time: [], wip: [], throughput: [] };
                this.stationHistory = {};
            }
        } else {
            // Record current state
            const wip = this._currentWIP();
            const elapsed = this.clock - this.warmupTime;
            const throughput = elapsed > 0 ? (this.completedJobs.length / elapsed) * 3600 : 0;

            this.history.time.push(this.clock - this.warmupTime);
            this.history.wip.push(wip);
            this.history.throughput.push(throughput);

            // Cost accounting (every 10s sim time)
            const intervalHours = 10 / 3600;
            // Labor cost: all non-quit operators cost money every tick
            const activeWorkers = this.operators.filter(o => o.status !== 'quit').length;
            const otWorkers = this.operators.filter(o => o.status !== 'quit' && o.overtimeHours > 0).length;
            const baseLaborCost = activeWorkers * this.costConfig.laborCostPerHour * intervalHours;
            const otPremiumCost = otWorkers * this.costConfig.laborCostPerHour * (this.costConfig.overtimePremium - 1) * intervalHours;
            this.costs.labor += baseLaborCost;
            this.costs.overtimeCost += otPremiumCost;
            // Holding cost: every WIP unit costs money per hour
            this.costs.holdingCost += wip * this.costConfig.holdingCostPerUnitHour * intervalHours;
            // Total
            this.costs.totalCost = this.costs.labor + this.costs.overtimeCost + this.costs.material + this.costs.scrapWaste + this.costs.holdingCost;

            // Per-station queue lengths
            for (const [id, stn] of this.stations) {
                if (stn.type === 'machine') {
                    if (!this.stationHistory[id]) this.stationHistory[id] = { name: stn.name, queue: [] };
                    this.stationHistory[id].queue.push(stn.queue.length);
                    stn.stats.queueSum += stn.queue.length;
                    stn.stats.queueSamples++;
                }
                // Record FG inventory snapshots
                if (stn.type === 'sink' && stn.sinkMode === 'fg_warehouse') {
                    stn.fgHistory.push({
                        time: this.clock - this.warmupTime,
                        inventory: { ...stn.fgInventory },
                    });
                }
            }
        }

        // Schedule next
        if (this.clock < this.warmupTime + this.runTime) {
            this.eventQueue.push({ time: this.clock + 10, type: 'record_stats' });
        }
    }

    _currentWIP() {
        let wip = 0;
        for (const [, stn] of this.stations) {
            if (stn.type === 'machine') {
                wip += stn.queue.length + (stn.currentJob ? 1 : 0) + (stn.blockedJob ? 1 : 0);
            }
        }
        return wip;
    }

    // Check blocked stations when downstream pulls
    _unblockUpstream(stnId) {
        const upstreams = this.reverseAdj.get(stnId) || [];
        for (const upId of upstreams) {
            const upStn = this.stations.get(upId);
            if (upStn && upStn.state === 'blocked' && upStn.blockedJob) {
                const targetStn = this.stations.get(stnId);
                if (targetStn && this._canAccept(stnId)) {
                    const job = upStn.blockedJob;
                    upStn.blockedJob = null;
                    upStn.stats.processed++;
                    this._updateStationTime(upStn, 'idle');
                    this._tryStartProcessing(upStn.id);
                    const transportTime = this._getTransportTime(upId, stnId);
                    if (transportTime > 0) {
                        this._requestTransport(job, stnId, transportTime);
                    } else {
                        targetStn.queue.push(job);
                        job.stationId = stnId;
                        this._tryStartProcessing(stnId);
                    }
                }
            }
        }
    }

    // Run modes
    runAnimated(speedFactor, onFrame) {
        this.running = true;
        const startReal = performance.now();
        const startClock = this.clock;

        const tick = () => {
            if (!this.running) return;
            if (this.paused) {
                onFrame(this.getState());
                animFrameId = requestAnimationFrame(tick);
                return;
            }
            const realElapsed = (performance.now() - startReal) / 1000;
            const targetTime = startClock + realElapsed * speedFactor;
            const endTime = this.warmupTime + this.runTime;

            let eventsProcessed = 0;
            while (this.eventQueue.size > 0 && this.eventQueue.peek().time <= targetTime && eventsProcessed < 500) {
                if (!this.processNextEvent()) { this.running = false; break; }
                eventsProcessed++;
                if (this.paused) break;
            }

            onFrame(this.getState());

            if (this.running && this.clock < endTime) {
                animFrameId = requestAnimationFrame(tick);
            } else {
                this.running = false;
                onFrame(this.getState(), true);
            }
        };
        animFrameId = requestAnimationFrame(tick);
    }

    runToCompletion() {
        const endTime = this.warmupTime + this.runTime;
        let safety = 0;
        while (this.eventQueue.size > 0 && this.clock < endTime && safety < 1000000) {
            if (!this.processNextEvent()) break;
            safety++;
        }
        return this.getResults();
    }

    step() {
        this.processNextEvent();
        return this.getState();
    }

    getState() {
        const elapsed = Math.max(0, this.clock - this.warmupTime);
        const throughput = elapsed > 0 ? (this.completedJobs.length / elapsed) * 3600 : 0;
        const wip = this._currentWIP();
        const avgLeadTime = this.completedJobs.length > 0
            ? this.completedJobs.reduce((s, j) => s + (j.completedAt - j.createdAt), 0) / this.completedJobs.length
            : 0;

        // Find bottleneck (highest utilization machine)
        let bottleneck = null;
        let maxUtil = 0;
        for (const [id, stn] of this.stations) {
            if (stn.type !== 'machine') continue;
            const total = stn.stats.processing + stn.stats.setup + stn.stats.down + stn.stats.starved + stn.stats.blocked + stn.stats.idle + stn.stats.onBreak;
            const util = total > 0 ? (stn.stats.processing + stn.stats.blocked) / total : 0;
            if (util > maxUtil) { maxUtil = util; bottleneck = stn; }
        }

        // Quality stats for live display
        let scrapped = 0, reworked = 0, processed = 0, changeovers = 0;
        for (const [id, stn] of this.stations) {
            if (stn.type !== 'machine') continue;
            scrapped += stn.stats.scrapped;
            reworked += stn.stats.reworked;
            processed += stn.stats.processed;
            changeovers += stn.stats.changeovers;
        }

        // FG inventory & demand stats
        const fgInventory = {};
        let totalStockouts = 0;
        let totalDemandFilled = 0;
        for (const [id, stn] of this.stations) {
            if (stn.type !== 'sink' || stn.sinkMode !== 'fg_warehouse') continue;
            for (const [pt, qty] of Object.entries(stn.fgInventory || {})) {
                fgInventory[pt] = (fgInventory[pt] || 0) + qty;
            }
            for (const [pt, cnt] of Object.entries(stn.stockouts || {})) {
                totalStockouts += cnt;
            }
            for (const [pt, cnt] of Object.entries(stn.demandFilled || {})) {
                totalDemandFilled += cnt;
            }
        }
        const serviceLevel = (totalDemandFilled + totalStockouts) > 0
            ? totalDemandFilled / (totalDemandFilled + totalStockouts) : 1;

        // Rush/due date metrics
        const onTimeDelivery = (this.onTimeCount + this.lateOrderCount) > 0
            ? this.onTimeCount / (this.onTimeCount + this.lateOrderCount) : 1;
        // Count WIP jobs that are currently past due
        let wipPastDue = 0;
        for (const j of this.jobs) {
            if (!j.completedAt && j.dueDate && this.clock > j.dueDate) wipPastDue++;
        }

        return {
            clock: this.clock,
            elapsed,
            throughput,
            wip,
            avgLeadTime,
            bottleneckName: bottleneck ? bottleneck.name : '—',
            bottleneckUtil: maxUtil,
            stations: Array.from(this.stations.values()),
            completedCount: this.completedJobs.length,
            scrapped, reworked, processed, changeovers,
            yieldRate: processed > 0 ? (processed - scrapped) / processed : 1,
            fgInventory,
            totalStockouts,
            totalDemandFilled,
            serviceLevel,
            // Rush order metrics
            rushOrderCount: this.rushOrderCount,
            lateOrderCount: this.lateOrderCount,
            onTimeCount: this.onTimeCount,
            onTimeDelivery,
            promotedToRush: this.promotedToRush,
            wipPastDue,
            // Escaped defect metrics
            escapedDefects: this.escapedDefects,
            detectedDownstream: this.detectedDownstream,
            customerReturns: this.customerReturns,
            // WIP aging
            expiredWIP: this.expiredWIP,
            // ECO
            ecoCount: this.ecoCount,
            ecoScrappedWIP: this.ecoScrappedWIP,
            // Management
            managementInterventions: this.management.interventionCount,
            managementLog: this.management.policyLog.slice(-5),
            // Overtime
            overtimeActive: this.overtimeActive,
            overtimeShifts: this.overtimeShifts,
            totalOTHours: this.totalOTHours,
            // Cost accounting
            costs: { ...this.costs },
            costPerUnit: this.completedJobs.length > 0
                ? this.costs.totalCost / this.completedJobs.length : 0,
            // AGV fleet metrics
            agvFleet: {
                size: this.agvFleet.size,
                available: this.agvFleet.available,
                queueLength: this.agvFleet.transportQueue.length,
                avgWaitTime: this.agvFleet.tripsCompleted > 0
                    ? this.agvFleet.totalWaitTime / this.agvFleet.tripsCompleted : 0,
                tripsCompleted: this.agvFleet.tripsCompleted,
            },
            // Utility system metrics
            utilitySystems: this.utilitySystems.map(u => ({
                id: u.id,
                name: u.name,
                state: u.state,
                failureCount: u.failureCount,
                totalDowntime: u.totalDowntime + (u.downSince ? this.clock - u.downSince : 0),
                affectedCount: u.affectedMachines.length,
            })),
            // Customer behavior metrics
            customerSatisfaction: this.customerSatisfaction,
            totalRevenue: this.totalRevenue,
            totalReturnCost: this.totalReturnCost,
            netRevenue: this.totalRevenue - this.totalReturnCost - this.costs.totalCost,
            customerLostOrders: this.customerLostOrders,
            // Maintenance crew metrics
            maintenanceCrew: {
                size: this.maintenanceCrew.size,
                available: this.maintenanceCrew.available,
                queueLength: this.maintenanceCrew.repairQueue.length,
                avgWaitTime: this.maintenanceCrew.repairsCompleted > 0
                    ? this.maintenanceCrew.totalWaitTime / this.maintenanceCrew.repairsCompleted : 0,
                repairsCompleted: this.maintenanceCrew.repairsCompleted,
            },
        };
    }

    getResults() {
        const state = this.getState();
        const stationUtils = {};
        for (const [id, stn] of this.stations) {
            if (stn.type !== 'machine') continue;
            const total = stn.stats.processing + stn.stats.setup + stn.stats.down + stn.stats.starved + stn.stats.blocked + stn.stats.idle + stn.stats.onBreak;
            stationUtils[id] = {
                name: stn.name,
                processing: total > 0 ? stn.stats.processing / total : 0,
                setup: total > 0 ? stn.stats.setup / total : 0,
                down: total > 0 ? stn.stats.down / total : 0,
                starved: total > 0 ? stn.stats.starved / total : 0,
                blocked: total > 0 ? stn.stats.blocked / total : 0,
                idle: total > 0 ? stn.stats.idle / total : 0,
                onBreak: total > 0 ? stn.stats.onBreak / total : 0,
                processed: stn.stats.processed,
                scrapped: stn.stats.scrapped,
                reworked: stn.stats.reworked,
                changeovers: stn.stats.changeovers,
                avgQueue: stn.stats.queueSamples > 0 ? stn.stats.queueSum / stn.stats.queueSamples : 0,
            };
        }

        const leadTimes = this.completedJobs.map(j => j.completedAt - j.createdAt);

        // Aggregate quality stats
        let totalScrapped = 0, totalReworked = 0, totalProcessed = 0, totalChangeovers = 0;
        for (const [id, stn] of this.stations) {
            if (stn.type !== 'machine') continue;
            totalScrapped += stn.stats.scrapped;
            totalReworked += stn.stats.reworked;
            totalProcessed += stn.stats.processed;
            totalChangeovers += stn.stats.changeovers;
        }
        const yieldRate = totalProcessed > 0 ? (totalProcessed - totalScrapped) / totalProcessed : 1;

        // Product mix breakdown from completed jobs
        const productCounts = {};
        for (const j of this.completedJobs) {
            const pt = j.productType || 'A';
            productCounts[pt] = (productCounts[pt] || 0) + 1;
        }

        // FG inventory & service level
        const fgInventory = {};
        const stockoutsByProduct = {};
        const demandFilledByProduct = {};
        let totalStockoutCount = 0, totalFilledCount = 0;
        for (const [id, stn] of this.stations) {
            if (stn.type !== 'sink' || stn.sinkMode !== 'fg_warehouse') continue;
            for (const [pt, qty] of Object.entries(stn.fgInventory || {})) {
                fgInventory[pt] = (fgInventory[pt] || 0) + qty;
            }
            for (const [pt, cnt] of Object.entries(stn.stockouts || {})) {
                stockoutsByProduct[pt] = (stockoutsByProduct[pt] || 0) + cnt;
                totalStockoutCount += cnt;
            }
            for (const [pt, cnt] of Object.entries(stn.demandFilled || {})) {
                demandFilledByProduct[pt] = (demandFilledByProduct[pt] || 0) + cnt;
                totalFilledCount += cnt;
            }
        }
        const serviceLevel = (totalFilledCount + totalStockoutCount) > 0
            ? totalFilledCount / (totalFilledCount + totalStockoutCount) : 1;

        return {
            throughput: state.throughput,
            avg_wip: state.wip,
            avg_lead_time: state.avgLeadTime,
            bottleneck_station_name: state.bottleneckName,
            bottleneck_utilization: state.bottleneckUtil,
            completed_count: this.completedJobs.length,
            total_scrapped: totalScrapped,
            total_reworked: totalReworked,
            total_changeovers: totalChangeovers,
            yield_rate: yieldRate,
            product_mix: productCounts,
            fg_inventory: fgInventory,
            stockouts_by_product: stockoutsByProduct,
            demand_filled_by_product: demandFilledByProduct,
            service_level: serviceLevel,
            total_stockouts: totalStockoutCount,
            // Rush order metrics
            rush_order_count: this.rushOrderCount,
            late_order_count: this.lateOrderCount,
            on_time_count: this.onTimeCount,
            on_time_delivery: state.onTimeDelivery,
            promoted_to_rush: this.promotedToRush,
            // Escaped defect metrics
            escaped_defects: this.escapedDefects,
            detected_downstream: this.detectedDownstream,
            customer_returns: this.customerReturns,
            // Customer behavior
            customer_satisfaction: this.customerSatisfaction,
            total_revenue: this.totalRevenue,
            total_return_cost: this.totalReturnCost,
            net_revenue: this.totalRevenue - this.totalReturnCost - this.costs.totalCost,
            customer_lost_orders: this.customerLostOrders,
            // ECO
            eco_count: this.ecoCount,
            eco_scrapped_wip: this.ecoScrappedWIP,
            // Cost accounting
            costs: { ...this.costs },
            cost_per_unit: this.completedJobs.length > 0
                ? this.costs.totalCost / this.completedJobs.length : 0,
            station_utilizations: stationUtils,
            lead_times: leadTimes,
            history: this.history,
            station_history: this.stationHistory,
            decision_log: this.decisionLog,
            mission_alerts: this.missionAlerts,
        };
    }

    // ===== PLAYER INTERVENTION METHODS =====
    _logDecision(action, target, detail) {
        let p = 0, s = 0;
        for (const [, stn] of this.stations) {
            if (stn.type === 'machine') { p += stn.stats.processed; s += stn.stats.scrapped; }
        }
        this.decisionLog.push({
            time: this.clock, action, target, detail,
            stateSnapshot: {
                wip: this._currentWIP(),
                throughput: this.completedJobs.length > 0
                    ? (this.completedJobs.length / Math.max(1, this.clock - this.warmupTime)) * 3600 : 0,
                yield: p > 0 ? (p - s) / p : 1,
            },
        });
    }

    interventionReassignOperator(operatorId, toStationId) {
        const op = this.operators.find(o => o.id === operatorId);
        if (!op) return false;
        if (op.assignedTo) {
            const prevStn = this.stations.get(op.assignedTo);
            if (prevStn && prevStn.currentJob) {
                this._enqueueJob(prevStn, prevStn.currentJob);
                prevStn.currentJob = null;
                this._updateStationTime(prevStn, 'starved');
            }
            op.status = 'available';
            op.assignedTo = null;
        }
        this._logDecision('reassign_operator', toStationId, `Reassigned ${op.name} to ${this.stations.get(toStationId)?.name || toStationId}`);
        this._tryStartProcessing(toStationId);
        return true;
    }

    interventionQuarantineStation(stationId) {
        const stn = this.stations.get(stationId);
        if (!stn || stn.type !== 'machine') return false;
        const scrapped = stn.queue.length;
        stn.queue = [];
        if (stn.currentJob) { this._releaseOperator(stn.id); stn.currentJob = null; }
        this._updateStationTime(stn, 'down');
        stn.stats.scrapped += scrapped;
        this._logDecision('quarantine', stationId, `Quarantined ${stn.name}: ${scrapped} WIP scrapped`);
        return true;
    }

    interventionAuthorizeOT(enable) {
        this.overtimeActive = enable;
        this._logDecision(enable ? 'authorize_ot' : 'cancel_ot', null, enable ? 'Authorized overtime' : 'Cancelled overtime');
        return true;
    }

    interventionChangeDispatchRule(stationId, rule) {
        const stn = this.stations.get(stationId);
        if (!stn) return false;
        const oldRule = stn.dispatchRule;
        stn.dispatchRule = rule;
        this._sortByDispatchRule(stn);
        this._logDecision('change_dispatch', stationId, `${stn.name}: ${oldRule} -> ${rule}`);
        return true;
    }

    interventionRequestMaintPriority(stationId) {
        const stn = this.stations.get(stationId);
        if (!stn || stn.state !== 'down') return false;
        const mc = this.maintenanceCrew;
        const idx = mc.repairQueue.findIndex(r => r.stationId === stationId);
        if (idx > 0) {
            const [item] = mc.repairQueue.splice(idx, 1);
            mc.repairQueue.unshift(item);
        }
        this._logDecision('maint_priority', stationId, `Prioritized maintenance for ${stn.name}`);
        return true;
    }

    interventionForceRepair(stationId) {
        const stn = this.stations.get(stationId);
        if (!stn || stn.state !== 'down') return false;
        const repairTime = stn.mttr ? this._sampleExponential(stn.mttr) * 0.5 : 60;
        this.eventQueue.push({ time: this.clock + repairTime, type: 'repair', stationId: stn.id });
        this._logDecision('force_repair', stationId, `Emergency repair for ${stn.name} (est. ${repairTime.toFixed(0)}s)`);
        return true;
    }

    interventionAdjustParam(stationId, param, value) {
        const stn = this.stations.get(stationId);
        if (!stn) return false;
        const oldVal = stn[param];
        stn[param] = value;
        this._logDecision('adjust_param', stationId, `${stn.name}.${param}: ${oldVal} -> ${value}`);
        return true;
    }

    interventionScrapSuspectWIP(stationId) {
        const stn = this.stations.get(stationId);
        if (!stn) return false;
        const count = stn.queue.length;
        stn.queue = [];
        stn.stats.scrapped += count;
        this._logDecision('scrap_wip', stationId, `Scrapped ${count} suspect WIP at ${stn.name}`);
        return true;
    }

    interventionStopMachine(stationId) {
        const stn = this.stations.get(stationId);
        if (!stn || stn.type !== 'machine') return false;
        if (stn.currentJob) {
            this._releaseOperator(stn.id);
            this._enqueueJob(stn, stn.currentJob);
            stn.currentJob = null;
        }
        this._updateStationTime(stn, 'down');
        stn._manualStop = true;
        this._logDecision('stop_machine', stationId, `Manually stopped ${stn.name}`);
        return true;
    }

    interventionStartMachine(stationId) {
        const stn = this.stations.get(stationId);
        if (!stn || stn.type !== 'machine' || !stn._manualStop) return false;
        stn._manualStop = false;
        this._updateStationTime(stn, 'idle');
        this._tryStartProcessing(stn.id);
        this._logDecision('start_machine', stationId, `Restarted ${stn.name}`);
        return true;
    }

    interventionAcknowledgeAlert(alertId) {
        const alert = this.missionAlerts.find(a => a.id === alertId);
        if (alert) { alert.acknowledged = true; alert.acknowledgedAt = this.clock; }
    }
}
