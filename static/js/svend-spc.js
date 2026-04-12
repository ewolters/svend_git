/**
 * svend-spc.js — Statistical Process Control charts for SVEND
 *
 * Load order: after svend-charts.js (uses SvendCharts.COLORS)
 * Used by: simulator.html, calculators.html (SPC Rare Events Lab)
 *
 * Provides:
 *   SvendSPC.iChartLimits()       — Compute I-chart control limits from observations
 *   SvendSPC.detectViolations()   — Western Electric Rules 1 & 4
 *   SvendSPC.updateIChart()       — Full I-chart update (limits + violations + Plotly)
 *   SvendSPC.cpk()                — Process capability Cpk
 *   SvendSPC.cp()                 — Process capability Cp
 */

const SvendSPC = (() => {
    'use strict';

    // d2 constant for n=2 (individual moving range)
    const D2_N2 = 1.128;

    /**
     * Compute I-chart control limits from observations.
     * Uses first `baselineCount` observations (default 20) to establish limits.
     * Sigma estimated from average moving range: σ = mR̄ / d2(n=2)
     *
     * @param {number[]} obs - Observation array
     * @param {number} [baselineCount=20] - Number of initial obs for baseline
     * @returns {{ mean: number, sigma: number, ucl: number, lcl: number }}
     */
    function iChartLimits(obs, baselineCount) {
        baselineCount = baselineCount || 20;
        const baseline = obs.slice(0, Math.min(baselineCount, obs.length));
        const mean = baseline.reduce((s, v) => s + v, 0) / baseline.length;

        let mrSum = 0;
        for (let i = 1; i < baseline.length; i++) {
            mrSum += Math.abs(baseline[i] - baseline[i - 1]);
        }
        const avgMR = baseline.length > 1 ? mrSum / (baseline.length - 1) : 0;
        const sigma = avgMR / D2_N2;

        return {
            mean,
            sigma,
            ucl: mean + 3 * sigma,
            lcl: mean - 3 * sigma,
        };
    }

    /**
     * Detect Western Electric Rule violations.
     * Rule 1: Any point beyond 3σ (UCL/LCL)
     * Rule 4: 8 consecutive points on same side of center line
     *
     * @param {number[]} obs - Observation array
     * @param {object} limits - { mean, ucl, lcl } from iChartLimits()
     * @returns {{ indices: number[], values: number[] }} - Violation points
     */
    function detectViolations(obs, limits) {
        const indices = [];
        const values = [];

        // Rule 1: beyond 3σ
        for (let i = 0; i < obs.length; i++) {
            if (obs[i] > limits.ucl || obs[i] < limits.lcl) {
                indices.push(i);
                values.push(obs[i]);
            }
        }

        // Rule 4: 8 consecutive same side
        if (obs.length >= 8) {
            for (let end = 7; end < obs.length; end++) {
                const window = obs.slice(end - 7, end + 1);
                if (window.every(v => v > limits.mean) || window.every(v => v < limits.mean)) {
                    if (!indices.includes(end)) {
                        indices.push(end);
                        values.push(obs[end]);
                    }
                }
            }
        }

        return { indices, values };
    }

    /**
     * Full I-chart update: compute limits, detect violations, update Plotly chart.
     * Expects the chart to already have 5 traces:
     *   [0] observations, [1] UCL, [2] CL, [3] LCL, [4] violation markers
     *
     * @param {string} chartId - Plotly div ID
     * @param {number[]} obs - Observation array
     * @param {number} [baselineCount=20]
     */
    function updateIChart(chartId, obs, baselineCount) {
        if (obs.length < 3) return;

        const limits = iChartLimits(obs, baselineCount);
        const violations = detectViolations(obs, limits);

        const n = obs.length;
        const uclLine = Array(n).fill(limits.ucl);
        const clLine = Array(n).fill(limits.mean);
        const lclLine = Array(n).fill(limits.lcl);

        Plotly.update(chartId, {
            y: [obs, uclLine, clLine, lclLine, violations.values],
            x: [undefined, undefined, undefined, undefined, violations.indices],
        }, {}, [0, 1, 2, 3, 4]);
    }

    /**
     * Process capability Cpk = min((USL - μ) / 3σ, (μ - LSL) / 3σ)
     * @param {number} mean - Process mean
     * @param {number} stddev - Process standard deviation
     * @param {number} usl - Upper spec limit
     * @param {number} lsl - Lower spec limit
     * @returns {number}
     */
    function cpk(mean, stddev, usl, lsl) {
        if (stddev <= 0) return Infinity;
        return Math.min((usl - mean) / (3 * stddev), (mean - lsl) / (3 * stddev));
    }

    /**
     * Process capability Cp = (USL - LSL) / 6σ
     * @param {number} stddev - Process standard deviation
     * @param {number} usl - Upper spec limit
     * @param {number} lsl - Lower spec limit
     * @returns {number}
     */
    function cp(stddev, usl, lsl) {
        if (stddev <= 0) return Infinity;
        return (usl - lsl) / (6 * stddev);
    }

    // ═══════════════════════════════════════════════════════════════
    //  PUBLIC API
    // ═══════════════════════════════════════════════════════════════

    return {
        iChartLimits,
        detectViolations,
        updateIChart,
        cpk,
        cp,
    };
})();
