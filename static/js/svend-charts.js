/**
 * svend-charts.js — Shared Plotly chart utilities for SVEND operations tools
 *
 * Load order: (none — standalone module)
 * Used by: calculators.html, simulator.html
 *
 * Provides:
 *   SvendCharts.LAYOUT        — Base Plotly layout (theme-aware, transparent)
 *   SvendCharts.LIVE_LAYOUT   — Compact layout for live/streaming charts
 *   SvendCharts.CONFIG        — Default Plotly config (responsive, no toolbar)
 *   SvendCharts.COLORS        — Standard trace color palette
 *   SvendCharts.plot()        — Plotly.newPlot wrapper with defaults
 *   SvendCharts.update()      — Plotly.react wrapper with defaults
 *   SvendCharts.gauge()       — Create a gauge indicator chart
 */

const SvendCharts = (() => {
    'use strict';

    // ═══════════════════════════════════════════════════════════════
    //  THEME-AWARE CONSTANTS
    // ═══════════════════════════════════════════════════════════════

    const FONT_COLOR = '#9aaa9a';
    const GRID_COLOR = 'rgba(255,255,255,0.05)';
    const GRID_COLOR_MEDIUM = 'rgba(255,255,255,0.07)';

    /** Base layout for standard charts */
    const LAYOUT = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: FONT_COLOR, size: 10 },
        margin: { t: 10, b: 40, l: 50, r: 50 },
        xaxis: { gridcolor: GRID_COLOR_MEDIUM },
        yaxis: { gridcolor: GRID_COLOR_MEDIUM },
        showlegend: true,
        legend: { orientation: 'h', y: -0.2, font: { size: 9 } },
    };

    /** Compact layout for live/streaming charts */
    const LIVE_LAYOUT = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: FONT_COLOR, size: 9 },
        margin: { t: 28, b: 24, l: 36, r: 8 },
        xaxis: { title: '', gridcolor: GRID_COLOR },
        yaxis: { gridcolor: GRID_COLOR },
        legend: { orientation: 'h', y: -0.15, font: { size: 8 } },
        showlegend: false,
        height: 230,
        autosize: true,
    };

    /** Default Plotly config */
    const CONFIG = { responsive: true, displayModeBar: false };

    /** Standard trace colors (operations palette) */
    const COLORS = {
        green:   '#4a9f6e',
        red:     '#e74c3c',
        orange:  '#f59e0b',
        blue:    '#3b82f6',
        purple:  '#8b5cf6',
        indigo:  '#6366f1',
        gray:    '#6b7280',
        teal:    '#14b8a6',
        pink:    '#ec4899',
        // State colors (for utilization stacked bars)
        processing: '#4a9f6e',
        blocked:    '#e74c3c',
        starved:    '#f59e0b',
        down:       '#dc2626',
        setup:      '#8b5cf6',
        idle:       '#6b7280',
        onBreak:    '#6366f1',
    };

    // ═══════════════════════════════════════════════════════════════
    //  CHART HELPERS
    // ═══════════════════════════════════════════════════════════════

    /**
     * Create a new Plotly chart with sensible defaults.
     * @param {string} divId - DOM element ID
     * @param {Array} traces - Plotly trace objects
     * @param {object} [layoutOverrides] - Merged onto LAYOUT
     * @param {object} [configOverrides] - Merged onto CONFIG
     */
    function plot(divId, traces, layoutOverrides, configOverrides) {
        const layout = { ...LAYOUT, ...layoutOverrides };
        // Deep-merge axis configs
        if (layoutOverrides) {
            if (layoutOverrides.xaxis) layout.xaxis = { ...LAYOUT.xaxis, ...layoutOverrides.xaxis };
            if (layoutOverrides.yaxis) layout.yaxis = { ...LAYOUT.yaxis, ...layoutOverrides.yaxis };
            if (layoutOverrides.legend) layout.legend = { ...LAYOUT.legend, ...layoutOverrides.legend };
        }
        const config = { ...CONFIG, ...configOverrides };
        Plotly.newPlot(divId, traces, layout, config);
    }

    /**
     * Update an existing Plotly chart (react — full redraw if needed).
     * @param {string} divId - DOM element ID
     * @param {Array} traces - Plotly trace objects
     * @param {object} [layoutOverrides] - Merged onto LAYOUT
     */
    function update(divId, traces, layoutOverrides) {
        const layout = { ...LAYOUT, ...layoutOverrides };
        if (layoutOverrides) {
            if (layoutOverrides.xaxis) layout.xaxis = { ...LAYOUT.xaxis, ...layoutOverrides.xaxis };
            if (layoutOverrides.yaxis) layout.yaxis = { ...LAYOUT.yaxis, ...layoutOverrides.yaxis };
            if (layoutOverrides.legend) layout.legend = { ...LAYOUT.legend, ...layoutOverrides.legend };
        }
        Plotly.react(divId, traces, layout, CONFIG);
    }

    /**
     * Create a Plotly gauge indicator.
     * @param {string} divId - DOM element ID
     * @param {number} value - Current value
     * @param {string} title - Gauge title
     * @param {object} [opts] - { min, max, suffix, thresholds: [{ range, color }] }
     */
    function gauge(divId, value, title, opts) {
        opts = opts || {};
        const trace = {
            type: 'indicator',
            mode: 'gauge+number',
            value: value,
            number: { suffix: opts.suffix || '' },
            gauge: {
                axis: {
                    range: [opts.min || 0, opts.max || 100],
                    tickcolor: FONT_COLOR,
                },
                bar: { color: COLORS.green },
                bgcolor: 'rgba(255,255,255,0.05)',
                steps: opts.thresholds || [],
            },
        };
        const layout = {
            paper_bgcolor: 'transparent',
            font: { color: FONT_COLOR, size: 11 },
            margin: { t: 30, b: 10, l: 30, r: 30 },
            title: { text: title, font: { size: 12 } },
            height: opts.height || 200,
        };
        Plotly.newPlot(divId, [trace], layout, CONFIG);
    }

    // ═══════════════════════════════════════════════════════════════
    //  PUBLIC API
    // ═══════════════════════════════════════════════════════════════

    return {
        LAYOUT,
        LIVE_LAYOUT,
        CONFIG,
        COLORS,
        plot,
        update,
        gauge,
    };
})();
