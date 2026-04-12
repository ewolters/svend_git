/**
 * VSM API Module
 * All server communication for Value Stream Maps.
 *
 * MIGRATION: Extracted from templates/vsm.html lines 1221-1434
 * Replaces raw fetch() calls with SvApi wrapper.
 * Old template still uses inline fetch() — this is the migration target.
 *
 * Dependencies: SvApi (sv-api.js)
 */

const VsmApi = (function() {
    'use strict';

    /**
     * List user's VSMs.
     * @param {Object} opts - {project_id, status}
     * @returns {Promise<Array>} VSM list
     */
    async function list(opts = {}) {
        const params = new URLSearchParams();
        if (opts.project_id) params.set('project_id', opts.project_id);
        if (opts.status) params.set('status', opts.status);
        const qs = params.toString();
        const data = await SvApi.get('/api/vsm/' + (qs ? '?' + qs : ''));
        return data.maps || [];
    }

    /**
     * Get a single VSM with full detail.
     * @param {string} id - VSM UUID
     * @returns {Promise<Object>} VSM data
     */
    async function get(id) {
        const data = await SvApi.get('/api/vsm/' + id + '/');
        return data.vsm || data;
    }

    /**
     * Create a new VSM.
     * @param {Object} payload - {name, product_family, customer_demand, project_id}
     * @returns {Promise<Object>} Created VSM
     */
    async function create(payload) {
        return SvApi.post('/api/vsm/create/', payload);
    }

    /**
     * Update VSM metadata + state.
     * @param {string} id - VSM UUID
     * @param {Object} payload - full VSM state
     * @returns {Promise<Object>}
     */
    async function update(id, payload) {
        return SvApi.put('/api/vsm/' + id + '/update/', payload);
    }

    /**
     * Delete a VSM.
     * @param {string} id
     */
    async function remove(id) {
        return SvApi.del('/api/vsm/' + id + '/');
    }

    /**
     * Add or update a process step.
     * @param {string} vsmId
     * @param {Object} step - {name, cycle_time, changeover, uptime, operators, batch_size, scrap, x, y, work_center_id}
     */
    async function saveStep(vsmId, step) {
        return SvApi.post('/api/vsm/' + vsmId + '/process-step/', step);
    }

    /**
     * Add inventory element.
     * @param {string} vsmId
     * @param {Object} inv - {name, days_of_supply, type, x, y}
     */
    async function addInventory(vsmId, inv) {
        return SvApi.post('/api/vsm/' + vsmId + '/inventory/', inv);
    }

    /**
     * Update inventory element.
     * @param {string} vsmId
     * @param {string} invId
     * @param {Object} inv
     */
    async function updateInventory(vsmId, invId, inv) {
        return SvApi.put('/api/vsm/' + vsmId + '/inventory/' + invId + '/', inv);
    }

    /**
     * Delete inventory element.
     * @param {string} vsmId
     * @param {string} invId
     */
    async function deleteInventory(vsmId, invId) {
        return SvApi.del('/api/vsm/' + vsmId + '/inventory/' + invId + '/');
    }

    /**
     * Add kaizen burst.
     * @param {string} vsmId
     * @param {Object} burst - {text, near_step, x, y}
     */
    async function addKaizen(vsmId, burst) {
        return SvApi.post('/api/vsm/' + vsmId + '/kaizen/', burst);
    }

    /**
     * Generate future state proposal.
     * @param {string} vsmId
     */
    async function futureState(vsmId) {
        return SvApi.get('/api/vsm/' + vsmId + '/future-state/');
    }

    /**
     * Compare current vs future state.
     * @param {string} vsmId
     */
    async function compare(vsmId) {
        return SvApi.get('/api/vsm/' + vsmId + '/compare/');
    }

    /**
     * Generate CI proposals (AI-driven).
     * @param {string} vsmId
     * @param {Object} opts - {focus_area, constraints}
     */
    async function generateProposals(vsmId, opts = {}) {
        return SvApi.post('/api/vsm/' + vsmId + '/generate-proposals/', opts);
    }

    /**
     * Approve a proposal → Hoshin.
     * @param {string} vsmId
     * @param {Object} proposal - {burst_id, title, savings_target, calc_method, project_type}
     */
    async function approveProposal(vsmId, proposal) {
        return SvApi.post('/api/vsm/' + vsmId + '/approve-proposal/', proposal);
    }

    /**
     * Promote VSM to Hoshin Kanri.
     * @param {string} vsmId
     */
    async function promoteToHoshin(vsmId) {
        return SvApi.post('/api/hoshin/vsm/' + vsmId + '/promote/', {});
    }

    /**
     * Load user projects for the project selector.
     * @returns {Promise<Array>}
     */
    async function listProjects() {
        return SvApi.get('/api/core/projects/');
    }

    return {
        list,
        get,
        create,
        update,
        remove,
        saveStep,
        addInventory,
        updateInventory,
        deleteInventory,
        addKaizen,
        futureState,
        compare,
        generateProposals,
        approveProposal,
        promoteToHoshin,
        listProjects,
    };
})();
