/**
 * sv-modal.js — Shared modal/dialog system for SVEND templates.
 *
 * Works with sv-modal-overlay + sv-modal from svend-widgets.css.
 *
 * Usage:
 *   // Open with HTML content
 *   SvModal.open('modal-id');
 *   SvModal.close('modal-id');
 *
 *   // Create programmatically
 *   const id = SvModal.create({ title: 'New Signal', body: '<form>...</form>', actions: [...] });
 *   SvModal.open(id);
 *
 *   // Confirm dialog
 *   const confirmed = await SvModal.confirm('Delete this signal?', { danger: true });
 *
 *   // Escape key closes topmost modal
 */

(function(global) {
    'use strict';

    let _counter = 0;

    function open(id) {
        const overlay = document.getElementById(id);
        if (overlay) overlay.classList.add('open');
    }

    function close(id) {
        const overlay = document.getElementById(id);
        if (overlay) overlay.classList.remove('open');
    }

    function create(opts) {
        opts = opts || {};
        const id = opts.id || ('sv-modal-' + (++_counter));

        const overlay = document.createElement('div');
        overlay.className = 'sv-modal-overlay';
        overlay.id = id;
        overlay.addEventListener('click', function(e) {
            if (e.target === overlay) close(id);
        });

        const modal = document.createElement('div');
        modal.className = 'sv-modal' + (opts.wide ? ' sv-modal-wide' : '') + (opts.narrow ? ' sv-modal-narrow' : '');

        if (opts.title) {
            const h3 = document.createElement('h3');
            h3.textContent = opts.title;
            modal.appendChild(h3);
        }

        if (opts.body) {
            const body = document.createElement('div');
            if (typeof opts.body === 'string') body.innerHTML = opts.body;
            else body.appendChild(opts.body);
            modal.appendChild(body);
        }

        if (opts.actions && opts.actions.length) {
            const actions = document.createElement('div');
            actions.className = 'sv-modal-actions';
            opts.actions.forEach(function(a) {
                const btn = document.createElement('button');
                btn.className = 'sv-btn ' + (a.className || '');
                btn.textContent = a.label;
                if (a.onclick) btn.addEventListener('click', a.onclick);
                actions.appendChild(btn);
            });
            modal.appendChild(actions);
        }

        overlay.appendChild(modal);
        document.body.appendChild(overlay);
        return id;
    }

    function confirm(message, opts) {
        opts = opts || {};
        return new Promise(function(resolve) {
            const id = create({
                title: opts.title || 'Confirm',
                body: '<p style="font-size:13px;color:var(--text-primary);margin:0;">' + (typeof esc === 'function' ? esc(message) : message) + '</p>',
                actions: [
                    { label: 'Cancel', className: 'sv-btn-ghost', onclick: function() { close(id); destroy(id); resolve(false); } },
                    { label: opts.confirmLabel || 'Confirm', className: opts.danger ? 'sv-btn-danger' : 'sv-btn-primary', onclick: function() { close(id); destroy(id); resolve(true); } },
                ],
            });
            open(id);
        });
    }

    function destroy(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    // Escape key closes topmost open modal
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.sv-modal-overlay.open');
            if (modals.length) {
                e.preventDefault();
                close(modals[modals.length - 1].id);
            }
        }
    });

    global.SvModal = {
        open: open,
        close: close,
        create: create,
        confirm: confirm,
        destroy: destroy,
    };

})(window);
