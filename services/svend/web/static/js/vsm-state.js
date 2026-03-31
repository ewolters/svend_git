/**
 * VSM State Module
 * Global state, icons, URL parsing.
 * MIGRATION: Extracted from templates/vsm.html
 */

// =============================================================================
// SVG Icon Paths (Svend style: stroke-based, 24x24 viewBox)
// =============================================================================
const VSM_ICONS = {
    customer: '<circle cx="9" cy="7" r="4"/><path d="M2 21v-2a4 4 0 0 1 4-4h6a4 4 0 0 1 4 4v2"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/>',
    supplier: '<path d="M4 21V11l5 3v-3l5 3V6h6v15"/><line x1="2" y1="21" x2="22" y2="21"/>',
    queue: '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    transport: '<rect x="1" y="3" width="15" height="13" rx="1"/><path d="M16 8h4l3 3v5h-7V8z"/><circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/>',
    batch: '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>'
};

