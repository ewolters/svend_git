/**
 * sv-api.js — Shared API client for SVEND templates.
 *
 * Replaces per-template fetch wrappers. Uses svCsrf() from base_app.html.
 *
 * Usage:
 *   const data = await SvApi.get('/api/loop/signals/');
 *   const result = await SvApi.post('/api/loop/signals/', { title: '...' });
 *   const result = await SvApi.put('/api/loop/signals/123/', { ... });
 *   const result = await SvApi.del('/api/loop/signals/123/');
 *
 * Error handling:
 *   try { ... } catch (e) { console.error(e.status, e.body); }
 */

(function(global) {
    'use strict';

    class ApiError extends Error {
        constructor(status, body) {
            super(`API error ${status}`);
            this.status = status;
            this.body = body;
        }
    }

    async function request(url, method, body) {
        const opts = {
            method: method,
            headers: { 'X-CSRFToken': typeof svCsrf === 'function' ? svCsrf() : '' },
        };
        if (body !== undefined) {
            opts.headers['Content-Type'] = 'application/json';
            opts.body = JSON.stringify(body);
        }
        const resp = await fetch(url, opts);
        if (!resp.ok) {
            let errBody = null;
            try { errBody = await resp.json(); } catch (_) { /* empty */ }
            throw new ApiError(resp.status, errBody);
        }
        const ct = resp.headers.get('content-type') || '';
        if (ct.includes('application/json')) return resp.json();
        return resp;
    }

    global.SvApi = {
        get:  function(url)       { return request(url, 'GET'); },
        post: function(url, body) { return request(url, 'POST', body); },
        put:  function(url, body) { return request(url, 'PUT', body); },
        patch: function(url, body) { return request(url, 'PATCH', body); },
        del:  function(url)       { return request(url, 'DELETE'); },
        ApiError: ApiError,
    };

})(window);
