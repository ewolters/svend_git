/**
 * sv-chart-controls.js — Glass cockpit chart control system.
 *
 * Sits between API and ForgeViz. Handles time range, filters,
 * aggregation, auto-refresh. Renders control bars that embed
 * alongside charts in instrument panels.
 *
 * Usage:
 *   const ctrl = SvChartCtrl.create(containerEl, {
 *       fetch: (params) => SvApi.get('/api/loop/signals/?' + new URLSearchParams(params)),
 *       transform: (data, params) => buildChartSpec(data, params),
 *       timeRanges: ['7d', '30d', '90d', '1y'],
 *       defaultRange: '30d',
 *       filters: [
 *           { key: 'source_type', label: 'Source', options: ['process', 'customer', 'monitoring'] },
 *           { key: 'severity', label: 'Severity', options: ['critical', 'warning', 'info'] },
 *       ],
 *       aggregation: ['daily', 'weekly', 'monthly'],
 *       refreshInterval: 0,  // seconds, 0 = off
 *   });
 *
 *   ctrl.refresh();           // manual refresh
 *   ctrl.setRange('90d');     // programmatic range change
 *   ctrl.setFilter('severity', ['critical']);
 *   ctrl.destroy();           // cleanup
 */

(function(global) {
    'use strict';

    function parseRange(range) {
        const now = new Date();
        const ms = { '24h': 864e5, '7d': 6048e5, '30d': 2592e6, '90d': 7776e6, '1y': 31536e6 };
        if (ms[range]) return { from: new Date(now - ms[range]).toISOString(), to: now.toISOString() };
        return {};
    }

    function create(container, opts) {
        opts = opts || {};
        const fetchFn = opts.fetch;
        const transformFn = opts.transform;
        const timeRanges = opts.timeRanges || [];
        const filters = opts.filters || [];
        const aggregations = opts.aggregation || [];
        const defaultRange = opts.defaultRange || '30d';
        const defaultAgg = opts.defaultAggregation || (aggregations[0] || '');

        let currentRange = defaultRange;
        let currentAgg = defaultAgg;
        let currentFilters = {};
        let refreshTimer = null;
        let chartEl = null;
        let controlEl = null;

        filters.forEach(f => { currentFilters[f.key] = []; });

        // ── Build DOM ──
        container.innerHTML = '';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.height = '100%';

        // Control bar — cockpit style, compact, embedded
        controlEl = document.createElement('div');
        controlEl.style.cssText = 'display:flex;align-items:center;gap:6px;padding:4px 6px;flex-wrap:wrap;border-bottom:1px solid var(--border);background:var(--bg-secondary);font-size:10px;min-height:24px;';
        container.appendChild(controlEl);

        // Chart area — needs min-height:0 for flex shrink + position for ForgeViz
        chartEl = document.createElement('div');
        chartEl.style.cssText = 'flex:1;min-height:0;overflow:hidden;position:relative;';
        container.appendChild(chartEl);

        function renderControls() {
            let html = '';

            // Time range — segment buttons
            if (timeRanges.length) {
                html += '<div class="sv-segment">';
                timeRanges.forEach(r => {
                    const active = r === currentRange ? ' active' : '';
                    html += '<button class="sv-segment-btn' + active + '" data-range="' + r + '">' + r + '</button>';
                });
                html += '</div>';
            }

            // Aggregation — segment buttons
            if (aggregations.length) {
                html += '<div class="sv-segment" style="margin-left:4px;">';
                aggregations.forEach(a => {
                    const active = a === currentAgg ? ' active' : '';
                    html += '<button class="sv-segment-btn' + active + '" data-agg="' + a + '">' + a.charAt(0).toUpperCase() + a.slice(1) + '</button>';
                });
                html += '</div>';
            }

            // Filters — dropdown style
            filters.forEach(f => {
                const active = currentFilters[f.key]?.length ? ' sv-badge-green' : '';
                html += '<div class="sv-dropdown" style="margin-left:4px;">';
                html += '<button class="sv-segment-btn" data-filter-toggle="' + f.key + '">' + f.label + (currentFilters[f.key]?.length ? ' (' + currentFilters[f.key].length + ')' : '') + '</button>';
                html += '<div class="sv-dropdown-menu" id="filter-menu-' + f.key + '">';
                html += '<div class="sv-dropdown-item" data-filter-clear="' + f.key + '" style="color:var(--text-dim);font-style:italic;">Clear</div>';
                html += '<div class="sv-dropdown-sep"></div>';
                f.options.forEach(o => {
                    const checked = (currentFilters[f.key] || []).includes(o);
                    html += '<div class="sv-dropdown-item" data-filter-key="' + f.key + '" data-filter-val="' + o + '">' +
                        '<span style="width:12px;text-align:center;font-size:10px;">' + (checked ? '&#x2713;' : '') + '</span> ' +
                        (typeof esc === 'function' ? esc(typeof SvFormat !== 'undefined' ? SvFormat.snakeToTitle(o) : o) : o) +
                        '</div>';
                });
                html += '</div></div>';
            });

            // Spacer + refresh indicator
            html += '<span style="margin-left:auto;color:var(--text-dim);" id="ctrl-status"></span>';

            controlEl.innerHTML = html;

            // Event handlers
            controlEl.querySelectorAll('[data-range]').forEach(btn => {
                btn.addEventListener('click', function() {
                    currentRange = this.dataset.range;
                    renderControls();
                    refresh();
                });
            });

            controlEl.querySelectorAll('[data-agg]').forEach(btn => {
                btn.addEventListener('click', function() {
                    currentAgg = this.dataset.agg;
                    renderControls();
                    refresh();
                });
            });

            controlEl.querySelectorAll('[data-filter-toggle]').forEach(btn => {
                btn.addEventListener('click', function(e) {
                    e.stopPropagation();
                    const menu = document.getElementById('filter-menu-' + this.dataset.filterToggle);
                    if (menu) menu.classList.toggle('open');
                });
            });

            controlEl.querySelectorAll('[data-filter-key]').forEach(item => {
                item.addEventListener('click', function(e) {
                    e.stopPropagation();
                    const key = this.dataset.filterKey;
                    const val = this.dataset.filterVal;
                    const arr = currentFilters[key] || [];
                    const idx = arr.indexOf(val);
                    if (idx >= 0) arr.splice(idx, 1); else arr.push(val);
                    currentFilters[key] = arr;
                    renderControls();
                    refresh();
                });
            });

            controlEl.querySelectorAll('[data-filter-clear]').forEach(item => {
                item.addEventListener('click', function(e) {
                    e.stopPropagation();
                    currentFilters[this.dataset.filterClear] = [];
                    renderControls();
                    refresh();
                });
            });

            // Close dropdowns on outside click
            document.addEventListener('click', function closeMenus() {
                controlEl.querySelectorAll('.sv-dropdown-menu.open').forEach(m => m.classList.remove('open'));
            });
        }

        function refresh() {
            const status = controlEl.querySelector('#ctrl-status');
            if (status) status.textContent = 'loading...';

            const params = { ...parseRange(currentRange) };
            if (currentAgg) params.aggregation = currentAgg;
            Object.keys(currentFilters).forEach(k => {
                if (currentFilters[k].length) params[k] = currentFilters[k].join(',');
            });

            fetchFn(params).then(data => {
                const spec = transformFn(data, params);
                if (spec && typeof ForgeViz !== 'undefined') {
                    ForgeViz.render(chartEl, spec, { toolbar: false });
                } else if (!spec) {
                    chartEl.innerHTML = '<div class="sv-empty" style="padding:20px;">No data for selected range</div>';
                }
                if (status) status.textContent = '';
            }).catch(err => {
                chartEl.innerHTML = '<div class="sv-empty" style="padding:20px;">Error loading data</div>';
                if (status) status.textContent = 'error';
            });
        }

        function setRefreshInterval(seconds) {
            if (refreshTimer) clearInterval(refreshTimer);
            if (seconds > 0) {
                refreshTimer = setInterval(refresh, seconds * 1000);
            }
        }

        // ── Init ──
        renderControls();
        refresh();
        if (opts.refreshInterval) setRefreshInterval(opts.refreshInterval);

        // ── Public API ──
        return {
            refresh: refresh,
            setRange: function(r) { currentRange = r; renderControls(); refresh(); },
            setFilter: function(key, vals) { currentFilters[key] = vals || []; renderControls(); refresh(); },
            setAggregation: function(a) { currentAgg = a; renderControls(); refresh(); },
            setRefreshInterval: setRefreshInterval,
            getParams: function() {
                const p = { ...parseRange(currentRange) };
                if (currentAgg) p.aggregation = currentAgg;
                Object.keys(currentFilters).forEach(k => { if (currentFilters[k].length) p[k] = currentFilters[k].join(','); });
                return p;
            },
            destroy: function() {
                if (refreshTimer) clearInterval(refreshTimer);
                container.innerHTML = '';
            },
        };
    }

    global.SvChartCtrl = { create: create };

})(window);
