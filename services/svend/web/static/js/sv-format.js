/**
 * sv-format.js — Shared formatting utilities for SVEND templates.
 *
 * Consolidates duplicated formatDate, number formatting, duration,
 * and display helpers used across templates.
 *
 * Usage:
 *   SvFormat.date('2026-03-30T14:00:00Z')           → "Mar 30, 2026"
 *   SvFormat.dateTime('2026-03-30T14:00:00Z')        → "Mar 30, 2026 2:00 PM"
 *   SvFormat.relative('2026-03-30T14:00:00Z')        → "2h" (uses timeAgo from base)
 *   SvFormat.number(12345.678, 2)                    → "12,345.68"
 *   SvFormat.pct(0.872, 1)                           → "87.2%"
 *   SvFormat.duration(3661)                          → "1h 1m"
 *   SvFormat.fileSize(1048576)                       → "1.0 MB"
 *   SvFormat.truncate('long text...', 40)            → "long text..."
 *   SvFormat.plural(3, 'signal', 'signals')          → "3 signals"
 *   SvFormat.snakeToTitle('source_type')             → "Source Type"
 */

(function(global) {
    'use strict';

    const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

    function date(iso) {
        if (!iso) return '—';
        const d = new Date(iso);
        return MONTHS[d.getMonth()] + ' ' + d.getDate() + ', ' + d.getFullYear();
    }

    function dateTime(iso) {
        if (!iso) return '—';
        const d = new Date(iso);
        return date(iso) + ' ' + d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    }

    function relative(iso) {
        return typeof timeAgo === 'function' ? timeAgo(iso) : date(iso);
    }

    function number(val, decimals) {
        if (val === null || val === undefined) return '—';
        decimals = decimals !== undefined ? decimals : 0;
        return Number(val).toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    }

    function pct(val, decimals) {
        if (val === null || val === undefined) return '—';
        decimals = decimals !== undefined ? decimals : 0;
        return (val * 100).toFixed(decimals) + '%';
    }

    function duration(seconds) {
        if (!seconds && seconds !== 0) return '—';
        seconds = Math.round(seconds);
        if (seconds < 60) return seconds + 's';
        if (seconds < 3600) return Math.floor(seconds / 60) + 'm';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        return h + 'h' + (m ? ' ' + m + 'm' : '');
    }

    function fileSize(bytes) {
        if (!bytes && bytes !== 0) return '—';
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + ' MB';
        return (bytes / 1073741824).toFixed(1) + ' GB';
    }

    function truncate(str, len) {
        if (!str) return '';
        len = len || 40;
        return str.length > len ? str.slice(0, len - 1) + '…' : str;
    }

    function plural(count, singular, pluralForm) {
        pluralForm = pluralForm || singular + 's';
        return count + ' ' + (count === 1 ? singular : pluralForm);
    }

    function snakeToTitle(str) {
        if (!str) return '';
        return str.replace(/_/g, ' ').replace(/\b\w/g, function(c) { return c.toUpperCase(); });
    }

    global.SvFormat = {
        date: date,
        dateTime: dateTime,
        relative: relative,
        number: number,
        pct: pct,
        duration: duration,
        fileSize: fileSize,
        truncate: truncate,
        plural: plural,
        snakeToTitle: snakeToTitle,
    };

})(window);
