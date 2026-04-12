/**
 * sv-toast.js — Notification toasts for SVEND.
 *
 * Usage:
 *   SvToast.success('Signal acknowledged');
 *   SvToast.error('Failed to save');
 *   SvToast.warn('Overdue commitment');
 *   SvToast.info('3 new signals');
 */

(function(global) {
    'use strict';

    let _container = null;

    function getContainer() {
        if (!_container) {
            _container = document.createElement('div');
            _container.className = 'sv-toast-container';
            document.body.appendChild(_container);
        }
        return _container;
    }

    function show(message, type, duration) {
        duration = duration || 4000;
        const toast = document.createElement('div');
        toast.className = 'sv-toast sv-toast-' + type;
        toast.textContent = message;
        getContainer().appendChild(toast);
        setTimeout(function() {
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.3s';
            setTimeout(function() { toast.remove(); }, 300);
        }, duration);
    }

    global.SvToast = {
        success: function(msg, dur) { show(msg, 'success', dur); },
        error:   function(msg, dur) { show(msg, 'error', dur); },
        warn:    function(msg, dur) { show(msg, 'warn', dur); },
        info:    function(msg, dur) { show(msg, 'info', dur); },
    };

})(window);
