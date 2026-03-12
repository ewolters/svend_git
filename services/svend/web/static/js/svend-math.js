/**
 * svend-math.js — Shared mathematical engines for SVEND operations tools
 *
 * Load order: (none — this is a foundation module)
 * Used by: calculators.html, simulator.html
 *
 * Provides:
 *   SvendMath.randn()           — Box-Muller standard normal
 *   SvendMath.sampleNormal()    — Normal with mean and CV
 *   SvendMath.sampleExponential() — Exponential distribution
 *   SvendMath.sampleWeibull()   — Weibull distribution
 *   SvendMath.seededRNG()       — Deterministic LCG for reproducible runs
 *   SvendMath.phi()             — Standard normal CDF
 *   SvendMath.phiPDF()          — Standard normal PDF
 *   SvendMath.phiInv()          — Inverse normal CDF (quantile)
 *   SvendMath.factorial()       — Integer factorial
 *   SvendMath.erlangC()         — Erlang C wait probability
 *   SvendMath.gcd()             — Greatest common divisor
 *   SvendMath.throughput()      — Throughput from completions and elapsed time
 *   SvendMath.littlesLaw()      — L = λW solver
 *   SvendMath.littlesCheck()    — Verify L = λW within tolerance
 *   SvendMath.oee()             — OEE from A, P, Q components
 *   SvendMath.bottleneck()      — Find highest-utilization station
 *   SvendMath.utilization()     — Station utilization from time buckets
 *   SvendMath.costAccounting()  — Aggregate cost categories
 *   SvendMath.formatCost()      — "$1.2k" / "$3.5M" formatter
 *   SvendMath.formatSimTime()   — Seconds to "T+H:MM:SS"
 */

const SvendMath = (() => {
    'use strict';

    // ═══════════════════════════════════════════════════════════════
    //  DISTRIBUTIONS
    // ═══════════════════════════════════════════════════════════════

    /** Box-Muller standard normal random (mean=0, σ=1) */
    function randn() {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    /** Normal sample with mean and coefficient of variation. Clamped to 0.1 minimum. */
    function sampleNormal(mean, cv) {
        if (cv <= 0) return mean;
        return Math.max(0.1, mean * (1 + cv * randn()));
    }

    /** Exponential sample: -mean × ln(1 - U) */
    function sampleExponential(mean) {
        return -mean * Math.log(1 - Math.random());
    }

    /** Weibull sample: eta = scale (characteristic life), beta = shape */
    function sampleWeibull(eta, beta) {
        if (beta <= 0 || eta <= 0) return Infinity;
        const u = Math.random();
        return eta * Math.pow(-Math.log(1 - u), 1 / beta);
    }

    /**
     * Create a seeded LCG random generator for reproducible simulations.
     * Returns a function that produces values in [0, 1).
     */
    function seededRNG(seed) {
        let rng = seed;
        return () => {
            rng = (rng * 9301 + 49297) % 233280;
            return rng / 233280;
        };
    }

    // ═══════════════════════════════════════════════════════════════
    //  STATISTICAL FUNCTIONS
    // ═══════════════════════════════════════════════════════════════

    /** Standard normal CDF (Abramowitz & Stegun approximation) */
    function phi(x) {
        const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
              a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        const sign = x < 0 ? -1 : 1;
        x = Math.abs(x) / Math.sqrt(2);
        const t = 1.0 / (1.0 + p * x);
        const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        return 0.5 * (1.0 + sign * y);
    }

    /** Standard normal PDF */
    function phiPDF(x) {
        return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
    }

    /** Inverse normal CDF (rational approximation) */
    function phiInv(p) {
        if (p <= 0) return -8;
        if (p >= 1) return 8;
        if (p < 0.5) return -phiInv(1 - p);
        const t = Math.sqrt(-2 * Math.log(1 - p));
        const c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
        const d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
        return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
    }

    /** Integer factorial */
    function factorial(n) {
        if (n <= 1) return 1;
        let r = 1;
        for (let i = 2; i <= n; i++) r *= i;
        return r;
    }

    /**
     * Erlang C: P(wait) for c agents handling traffic intensity A.
     * Returns 1 if system is unstable (agents <= traffic).
     */
    function erlangC(agents, traffic) {
        if (agents <= traffic) return 1;
        const c = agents;
        const A = traffic;
        let num = Math.pow(A, c) / factorial(c) * (c / (c - A));
        let denom = 0;
        for (let k = 0; k < c; k++) {
            denom += Math.pow(A, k) / factorial(k);
        }
        denom += num;
        return num / denom;
    }

    /** Greatest common divisor (Euclidean) */
    function gcd(a, b) {
        return b === 0 ? a : gcd(b, a % b);
    }

    // ═══════════════════════════════════════════════════════════════
    //  OPERATIONS ENGINES
    // ═══════════════════════════════════════════════════════════════

    /** Throughput in units/hour from completed count and elapsed seconds */
    function throughput(completedCount, elapsedSeconds) {
        return elapsedSeconds > 0 ? (completedCount / elapsedSeconds) * 3600 : 0;
    }

    /**
     * Little's Law solver: L = λW
     * @param {string} solveFor - 'wip', 'throughput', or 'leadtime'
     * @param {object} params - { wip, throughput, leadtime } (provide 2 of 3)
     * @returns {{ value: number, label: string, unit: string }}
     */
    function littlesLaw(solveFor, params) {
        if (solveFor === 'wip') {
            return { value: params.throughput * params.leadtime, label: 'WIP', unit: 'units' };
        } else if (solveFor === 'throughput') {
            return { value: params.wip / params.leadtime, label: 'Throughput', unit: 'units/hr' };
        } else {
            return { value: params.wip / params.throughput, label: 'Lead Time', unit: 'hours' };
        }
    }

    /**
     * Verify Little's Law: |predicted - actual| / actual < tolerance
     * @returns {{ predicted: number, actual: number, match: boolean }}
     */
    function littlesCheck(throughputPerHour, avgLeadTimeSec, actualWIP, tolerance) {
        tolerance = tolerance || 0.3;
        const lambda = throughputPerHour / 3600;
        const predicted = lambda * avgLeadTimeSec;
        const match = actualWIP > 0 ? Math.abs(predicted - actualWIP) / actualWIP < tolerance : true;
        return { predicted, actual: actualWIP, match };
    }

    /**
     * OEE decomposition
     * @param {number} plannedMin - Total planned production time (minutes)
     * @param {number} downtimeMin - Unplanned downtime (minutes)
     * @param {number} idealCycleMin - Ideal cycle time per unit (minutes)
     * @param {number} produced - Total units produced
     * @param {number} defects - Defective units
     * @returns {{ oee, availability, performance, quality, runtime, good }}
     */
    function oee(plannedMin, downtimeMin, idealCycleMin, produced, defects) {
        const runtime = plannedMin - downtimeMin;
        const good = produced - defects;
        const availability = runtime / plannedMin;
        const performance = Math.min(1, (produced * idealCycleMin / 60) / runtime);
        const quality = produced > 0 ? good / produced : 1;
        return {
            oee: availability * performance * quality,
            availability,
            performance,
            quality,
            runtime,
            good,
        };
    }

    /**
     * Station utilization from time-in-state buckets.
     * @param {object} stats - { processing, setup, down, starved, blocked, idle, onBreak }
     * @returns {number} utilization ratio (0-1)
     */
    function utilization(stats) {
        const total = (stats.processing || 0) + (stats.setup || 0) + (stats.down || 0) +
                      (stats.starved || 0) + (stats.blocked || 0) + (stats.idle || 0) +
                      (stats.onBreak || 0);
        return total > 0 ? ((stats.processing || 0) + (stats.blocked || 0)) / total : 0;
    }

    /**
     * Find bottleneck: station with highest utilization.
     * @param {Array} stations - [{ name, stats: { processing, setup, down, starved, blocked, idle, onBreak }}]
     * @returns {{ name: string, util: number } | null}
     */
    function bottleneck(stations) {
        let best = null;
        let maxUtil = 0;
        for (const stn of stations) {
            const u = utilization(stn.stats);
            if (u > maxUtil) {
                maxUtil = u;
                best = stn;
            }
        }
        return best ? { name: best.name, util: maxUtil } : null;
    }

    /**
     * Aggregate cost accounting.
     * @param {object} params
     * @returns {object} { labor, overtimeCost, material, scrapWaste, holdingCost, totalCost }
     */
    function costAccounting(params) {
        const labor = (params.workerCount || 0) * (params.laborCostPerHour || 0) * (params.hoursWorked || 0);
        const overtimeCost = (params.otWorkerCount || 0) * (params.laborCostPerHour || 0) *
                             ((params.overtimePremium || 1.5) - 1) * (params.otHours || 0);
        const material = params.materialCost || 0;
        const scrapWaste = params.scrapCost || 0;
        const holdingCost = (params.avgWIP || 0) * (params.holdingCostPerUnitHour || 0) * (params.hoursWorked || 0);
        return {
            labor,
            overtimeCost,
            material,
            scrapWaste,
            holdingCost,
            totalCost: labor + overtimeCost + material + scrapWaste + holdingCost,
        };
    }

    // ═══════════════════════════════════════════════════════════════
    //  FORMATTERS
    // ═══════════════════════════════════════════════════════════════

    /** Format cost value: "$1.2k", "$3.5M" */
    function formatCost(n) {
        if (n >= 1000000) return `$${(n / 1000000).toFixed(1)}M`;
        if (n >= 1000) return `$${(n / 1000).toFixed(1)}k`;
        return `$${n.toFixed(0)}`;
    }

    /** Format seconds as simulation time: "T+H:MM:SS" or "T+M:SS" */
    function formatSimTime(seconds) {
        if (seconds < 0) seconds = 0;
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        return h > 0
            ? `T+${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
            : `T+${m}:${String(s).padStart(2, '0')}`;
    }

    // ═══════════════════════════════════════════════════════════════
    //  PUBLIC API
    // ═══════════════════════════════════════════════════════════════

    return {
        // Distributions
        randn,
        sampleNormal,
        sampleExponential,
        sampleWeibull,
        seededRNG,

        // Statistical functions
        phi,
        phiPDF,
        phiInv,
        factorial,
        erlangC,
        gcd,

        // Operations engines
        throughput,
        littlesLaw,
        littlesCheck,
        oee,
        utilization,
        bottleneck,
        costAccounting,

        // Formatters
        formatCost,
        formatSimTime,
    };
})();
