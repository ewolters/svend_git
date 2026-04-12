/**
 * sv-canvas.js — Shared pan/zoom canvas for SVEND workspaces.
 *
 * Used by: whiteboard, VSM, any template with a pannable/zoomable surface.
 * Replaces duplicated updateTransform / updateCanvasTransform implementations.
 *
 * Usage:
 *   const canvas = SvCanvas.init(document.getElementById('canvas-container'), {
 *       minZoom: 0.1,
 *       maxZoom: 5,
 *       onTransform: (tx, ty, scale) => { ... },
 *   });
 *   canvas.fitContent(contentBBox);
 *   canvas.resetView();
 *   canvas.setZoom(1.5);
 *   canvas.panTo(x, y);
 */

(function(global) {
    'use strict';

    function init(container, opts) {
        opts = opts || {};
        const minZoom = opts.minZoom || 0.1;
        const maxZoom = opts.maxZoom || 5;
        const onTransform = opts.onTransform || null;

        let tx = 0, ty = 0, scale = 1;
        let isPanning = false, startX = 0, startY = 0, startTx = 0, startTy = 0;

        // Inner element that gets transformed
        const inner = container.querySelector('[data-sv-canvas-inner]') || container.firstElementChild;
        if (!inner) return null;

        function applyTransform() {
            inner.style.transform = 'translate(' + tx + 'px, ' + ty + 'px) scale(' + scale + ')';
            inner.style.transformOrigin = '0 0';
            if (onTransform) onTransform(tx, ty, scale);
        }

        // Pan
        container.addEventListener('pointerdown', function(e) {
            if (e.button !== 0) return; // left button only
            if (e.target.closest('[data-sv-no-pan]')) return; // opt-out elements
            isPanning = true;
            startX = e.clientX; startY = e.clientY;
            startTx = tx; startTy = ty;
            container.style.cursor = 'grabbing';
            container.setPointerCapture(e.pointerId);
        });

        container.addEventListener('pointermove', function(e) {
            if (!isPanning) return;
            tx = startTx + (e.clientX - startX);
            ty = startTy + (e.clientY - startY);
            applyTransform();
        });

        container.addEventListener('pointerup', function(e) {
            if (!isPanning) return;
            isPanning = false;
            container.style.cursor = '';
            container.releasePointerCapture(e.pointerId);
        });

        // Zoom (wheel)
        container.addEventListener('wheel', function(e) {
            e.preventDefault();
            const rect = container.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;

            const oldScale = scale;
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            scale = Math.max(minZoom, Math.min(maxZoom, scale * delta));

            // Zoom toward cursor
            tx = mx - (mx - tx) * (scale / oldScale);
            ty = my - (my - ty) * (scale / oldScale);
            applyTransform();
        }, { passive: false });

        // Pinch (touch)
        let lastTouchDist = 0;
        container.addEventListener('touchstart', function(e) {
            if (e.touches.length === 2) {
                lastTouchDist = Math.hypot(
                    e.touches[0].clientX - e.touches[1].clientX,
                    e.touches[0].clientY - e.touches[1].clientY
                );
            }
        }, { passive: true });

        container.addEventListener('touchmove', function(e) {
            if (e.touches.length === 2) {
                e.preventDefault();
                const dist = Math.hypot(
                    e.touches[0].clientX - e.touches[1].clientX,
                    e.touches[0].clientY - e.touches[1].clientY
                );
                const ratio = dist / lastTouchDist;
                scale = Math.max(minZoom, Math.min(maxZoom, scale * ratio));
                lastTouchDist = dist;
                applyTransform();
            }
        }, { passive: false });

        // Public API
        return {
            resetView: function() {
                tx = 0; ty = 0; scale = 1;
                applyTransform();
            },
            setZoom: function(z) {
                scale = Math.max(minZoom, Math.min(maxZoom, z));
                applyTransform();
            },
            panTo: function(x, y) {
                tx = -x * scale + container.clientWidth / 2;
                ty = -y * scale + container.clientHeight / 2;
                applyTransform();
            },
            fitContent: function(bbox) {
                if (!bbox) return;
                const cw = container.clientWidth;
                const ch = container.clientHeight;
                scale = Math.min(cw / (bbox.width + 40), ch / (bbox.height + 40), maxZoom);
                tx = (cw - bbox.width * scale) / 2 - bbox.x * scale;
                ty = (ch - bbox.height * scale) / 2 - bbox.y * scale;
                applyTransform();
            },
            getTransform: function() { return { tx: tx, ty: ty, scale: scale }; },
        };
    }

    global.SvCanvas = { init: init };

})(window);
