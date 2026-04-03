# DESIGN-TOKENS — Visual Design Specification

**Version:** 1.0
**Date:** 2026-04-03
**Status:** ACTIVE
**Scope:** All production templates in `/services/svend/web/templates/`

---

## Purpose

Canonical design decisions for the SVEND UI. Templates must conform to these tokens. ForgeRack (`~/forgerack/`) is exempt — it maintains its own skeuomorphic design system.

## Decision: Flat with Instrument Vocabulary

The main product uses flat design with sv-* device components (LEDs, readouts, meters, device chassis) for identity. No skeuomorphism (brushed grain, SVG feTurbulence, CRT phosphor bloom, physical depth simulation) in production templates. Skeuomorphism is reserved for ForgeRack only.

---

## 1. Border Radius

| Token | Value | Use |
|-------|-------|-----|
| `var(--radius-sm)` | `4px` | Inputs, buttons, badges, tooltips |
| `var(--radius-md)` | `6px` | Cards, panels, modals, dropdowns |
| `var(--radius-lg)` | `12px` | Large containers, hero sections |
| `9999px` | full | Pills, dots, LED indicators, avatars |

No other values. Templates must not hardcode border-radius in px.

## 2. Colors

**Rule:** Never hardcode hex or rgb values in templates. Use CSS variables from `base_app.html`.

| Purpose | Variable |
|---------|----------|
| Accent | `--accent-primary` |
| Accent dim fill | `--accent-primary-dim` |
| Accent border | `--accent-primary-border` |
| Secondary accents | `--accent-blue`, `--accent-purple`, `--accent-gold`, `--accent-orange` |
| Status green | `--sv-green` / `--success` |
| Status amber | `--sv-amber` / `--warning` |
| Status red | `--sv-red` / `--error` |
| Status blue | `--sv-blue` / `--info` |
| Background (deepest) | `--bg-primary` |
| Background (panels) | `--bg-secondary` |
| Background (elevated) | `--bg-tertiary` |
| Background (hover) | `--bg-hover` |
| Card surface | `--card-bg` |
| Text main | `--text-primary` |
| Text supporting | `--text-secondary` |
| Text labels/meta | `--text-dim` |
| Border standard | `--border` |
| Border emphasis | `--border-heavy` |

## 3. Typography

| Use | Font | Size | Weight |
|-----|------|------|--------|
| Body text | Inter (inherited) | 12-13px | 400-500 |
| Industrial labels | `var(--font-label)` | 9-10px, uppercase, letter-spacing 0.04-0.06em | 500-600 |
| Values/metrics | JetBrains Mono | 11-28px | 500-700 |
| UI controls | Inter (inherited) | 10-12px | 500 |

## 4. Hover Patterns

| Element type | Hover behavior |
|-------------|---------------|
| Cards, cells | `border-color: var(--accent-primary-border)` |
| Table rows, list items | `background: var(--bg-hover)` |
| Buttons (default) | `border-color: var(--accent-primary-border); color: var(--text-primary)` |
| Buttons (primary) | `opacity: 0.9` |
| Buttons (ghost) | `background: var(--bg-hover)` |
| Tabs, links | `color: var(--text-primary)` |

**Prohibited:** `transform: translateY()`, `box-shadow` changes on hover, custom hover animations.

## 5. Buttons

Use `sv-btn` variants from `svend-widgets.css` exclusively:

| Class | Visual | Use |
|-------|--------|-----|
| `.sv-btn` | Ghost outline | Default/secondary actions |
| `.sv-btn-primary` | Filled accent | Primary CTA (one per view) |
| `.sv-btn-danger` | Red outline | Destructive actions |
| `.sv-btn-ghost` | No border | Toolbar/icon buttons |
| `.sv-btn-sm` | Small size | Inline/compact |
| `.sv-btn-lg` | Large size | Modal CTAs |
| `.sv-btn-icon` | Square icon-only | Toolbar icons |

No template-specific button classes.

## 6. Cards & Containers

| Class | Use |
|-------|-----|
| `.sv-card` | Generic content container |
| `.sv-kpi-cell` | KPI metric (value + label) |
| `.sv-chart-panel` | Chart with title bar |
| `.sv-device` | Instrument device faceplate |

**Prohibited names:** `.panel`, `.stat-card`, `.summary-card`, `.info-box`, `.metric-card`, `.kpi-card`. Use the sv-* equivalents.

## 7. Device Components (Flat Instrument Vocabulary)

These sv-* components from `svend-widgets.css` stay — they provide identity without skeuomorphism:

- `sv-device` — chassis with head strip, body, sections
- `sv-device-screw` — decorative corner screws
- `sv-led` — status light indicators (green/amber/red/blue/accent)
- `sv-readout` — 7-segment style numeric display
- `sv-meter` — horizontal bar gauge
- `sv-knob` — labeled parameter input
- `sv-patch-bar` — data I/O connection indicators
- Device themes: `device-sentinel`, `forge-spc`, etc. for accent colors

## 8. Retired

- `kjerne-theme.css` — archived at `static/css/archive/`. Must not be linked from production templates.
- All `kj-*` CSS classes and `--kj-*` CSS variables — replaced by `sv-*` classes and base theme variables.
- Brushed grain (`repeating-linear-gradient` faux-grain, SVG `feTurbulence` filters) in production templates.
- `kj-panel`, `kj-panel-raised` with specular highlights and inset shadows.

## 9. Enforcement

- Pre-commit hook `check_css_conventions.py` scans for violations
- FE-001 §CSS Conventions references this document
- Template migration checklist (Object_271 `migration_plan.md`) includes design token conformance
