/**
 * svend-sim-core.js — Shared simulation infrastructure for SVEND
 *
 * Load order: after svend-math.js
 * Used by: calculators.html (8 mini-simulators), simulator.html (PlantDES)
 *
 * Provides:
 *   MinHeap              — Binary min-heap priority queue (keyed on .time)
 *   SimTicker            — Interval-based tick engine with speed control
 *   SimRunner            — requestAnimationFrame runner with real-time sync
 *   svendToast()         — Lightweight toast notification
 *   svendGetCsrf()       — Extract CSRF token from cookie
 */

// ═══════════════════════════════════════════════════════════════
//  MinHeap — Priority Queue for Discrete Event Simulation
// ═══════════════════════════════════════════════════════════════

/**
 * Binary min-heap ordered by item.time.
 * O(log n) push/pop, O(1) peek.
 */
class MinHeap {
    constructor() { this.data = []; }
    get size() { return this.data.length; }
    peek() { return this.data[0]; }
    clear() { this.data = []; }

    push(item) {
        this.data.push(item);
        let i = this.data.length - 1;
        while (i > 0) {
            const parent = (i - 1) >> 1;
            if (this.data[parent].time <= this.data[i].time) break;
            [this.data[parent], this.data[i]] = [this.data[i], this.data[parent]];
            i = parent;
        }
    }

    pop() {
        const top = this.data[0];
        const last = this.data.pop();
        if (this.data.length > 0) {
            this.data[0] = last;
            let i = 0;
            while (true) {
                let smallest = i;
                const l = 2 * i + 1, r = 2 * i + 2;
                if (l < this.data.length && this.data[l].time < this.data[smallest].time) smallest = l;
                if (r < this.data.length && this.data[r].time < this.data[smallest].time) smallest = r;
                if (smallest === i) break;
                [this.data[smallest], this.data[i]] = [this.data[i], this.data[smallest]];
                i = smallest;
            }
        }
        return top;
    }
}

// ═══════════════════════════════════════════════════════════════
//  SimTicker — Interval-Based Tick Engine
// ═══════════════════════════════════════════════════════════════

/**
 * Generic tick engine for mini-simulators (calculators).
 * Uses setInterval with configurable speed multiplier.
 *
 * Usage:
 *   const ticker = new SimTicker({
 *       intervalMs: 100,
 *       onTick: (tickCount) => { ... },
 *       onComplete: () => { ... },
 *       speedMultiplier: 1,
 *   });
 *   ticker.start();
 *   ticker.stop();
 *   ticker.reset();
 */
class SimTicker {
    constructor(opts) {
        this.intervalMs = opts.intervalMs || 100;
        this.onTick = opts.onTick;
        this.onComplete = opts.onComplete || (() => {});
        this.speedMultiplier = opts.speedMultiplier || 1;
        this._intervalId = null;
        this._tickCount = 0;
        this.running = false;
    }

    start() {
        if (this.running) return;
        this.running = true;
        this._intervalId = setInterval(() => {
            for (let i = 0; i < this.speedMultiplier; i++) {
                this._tickCount++;
                const shouldStop = this.onTick(this._tickCount);
                if (shouldStop) {
                    this.stop();
                    this.onComplete();
                    return;
                }
            }
        }, this.intervalMs);
    }

    stop() {
        this.running = false;
        if (this._intervalId !== null) {
            clearInterval(this._intervalId);
            this._intervalId = null;
        }
    }

    reset() {
        this.stop();
        this._tickCount = 0;
    }

    setSpeed(multiplier) {
        this.speedMultiplier = multiplier;
    }
}

// ═══════════════════════════════════════════════════════════════
//  SimRunner — requestAnimationFrame DES Runner
// ═══════════════════════════════════════════════════════════════

/**
 * Frame-synced simulation runner for full DES engines.
 * Keeps sim clock proportional to wall-clock time × speedFactor.
 *
 * Usage:
 *   const runner = new SimRunner({
 *       processEvent: () => { ... return false if more events },
 *       peekTime: () => nextEventTime,
 *       getClock: () => simClock,
 *       endTime: 7200,
 *       speedFactor: 10,
 *       maxEventsPerFrame: 500,
 *       onFrame: (state) => { ... },
 *       onComplete: (results) => { ... },
 *   });
 *   runner.start();
 *   runner.pause();
 *   runner.resume();
 */
class SimRunner {
    constructor(opts) {
        this.processEvent = opts.processEvent;
        this.peekTime = opts.peekTime;
        this.getClock = opts.getClock;
        this.endTime = opts.endTime || 7200;
        this.speedFactor = opts.speedFactor || 10;
        this.maxEventsPerFrame = opts.maxEventsPerFrame || 500;
        this.onFrame = opts.onFrame || (() => {});
        this.onComplete = opts.onComplete || (() => {});
        this._animFrameId = null;
        this._startReal = null;
        this._startClock = null;
        this.running = false;
        this.paused = false;
    }

    start() {
        this.running = true;
        this.paused = false;
        this._startReal = performance.now();
        this._startClock = this.getClock();
        this._tick();
    }

    pause() {
        this.paused = true;
        this.running = false;
        if (this._animFrameId !== null) {
            cancelAnimationFrame(this._animFrameId);
            this._animFrameId = null;
        }
    }

    resume() {
        if (!this.paused) return;
        this.paused = false;
        this.running = true;
        this._startReal = performance.now();
        this._startClock = this.getClock();
        this._tick();
    }

    _tick() {
        const now = performance.now();
        const realElapsed = (now - this._startReal) / 1000;
        const targetTime = this._startClock + realElapsed * this.speedFactor;

        let eventsProcessed = 0;
        while (eventsProcessed < this.maxEventsPerFrame) {
            const nextTime = this.peekTime();
            if (nextTime === undefined || nextTime > targetTime || nextTime > this.endTime) break;
            this.processEvent();
            eventsProcessed++;
        }

        this.onFrame();

        const clock = this.getClock();
        if (this.running && clock < this.endTime && this.peekTime() !== undefined) {
            this._animFrameId = requestAnimationFrame(() => this._tick());
        } else {
            this.running = false;
            this.onComplete();
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  SHARED UI UTILITIES
// ═══════════════════════════════════════════════════════════════

/**
 * Lightweight toast notification.
 * @param {string} message - Text to display
 * @param {number} [durationMs=3000] - Auto-dismiss after this many ms
 */
function svendToast(message, durationMs) {
    durationMs = durationMs || 3000;
    const existing = document.querySelector('.svend-toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'svend-toast';
    toast.style.cssText = `
        position: fixed; bottom: 20px; right: 20px; padding: 12px 20px;
        background: var(--accent, #4a9f6e); color: white; border-radius: 8px;
        font-size: 13px; font-weight: 500; z-index: 9999;
        animation: fadeInUp 0.3s ease;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), durationMs);
}

/**
 * Extract CSRF token from cookie.
 * @returns {string}
 */
function svendGetCsrf() {
    const cookie = document.cookie.split(';').find(c => c.trim().startsWith('csrftoken='));
    return cookie ? cookie.split('=')[1] : '';
}
