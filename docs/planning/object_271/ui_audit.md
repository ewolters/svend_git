# Object 271 — UI Audit

**Date:** 2026-03-29
**Agent:** S2 (Conservative)
**Scope:** All 45 user-facing templates. Styling patterns, consistency, "AI-built" indicators.

---

## Verdict: It's Better Than Expected

The UI is **not** bento-box AI-dashboard. It's professional, dense, intentional. The aesthetic is closer to RStudio/Minitab than to a ChatGPT-generated dashboard. Shadows are sparse, grids are semantic, typography hierarchy is strict.

But there are real issues that make it feel "assembled" rather than "designed as one system."

---

## The Real Problems

### 1. Border Radius Is Incoherent

Seven distinct values across templates:

| Value | Where | Feel |
|-------|-------|------|
| 3px | Base buttons, form inputs | Technical/sharp |
| 4px | Base `.card` class | Technical |
| 6px | Some panels, dropdowns | Transition |
| 8px | Some modals | Transition |
| 10px | Dashboard cards (loop, safety, hoshin, investigation) | Friendly |
| 12px | Main dashboard tool cards, project cards | Very friendly |
| 20px | Rare filter buttons | Pill-like |

**The problem:** Walking from the analysis workbench (3px everything) to the loop dashboard (10px everything) to the main dashboard (12px everything) feels like three different products. The user's eye notices even if they can't articulate it.

**Cockpit fix:** Pick ONE radius for interactive cards (suggest 6px — technical but not aggressive), ONE for elevated panels/modals (8px), and leave pills/badges fully rounded. Kill 10px and 12px on cards.

### 2. The Main Dashboard Is the Worst Offender

`dashboard.html` has the most "AI-built" feel:
- `repeat(auto-fill, minmax(280px, 1fr))` flowing grid of tool cards
- Cards lift on hover (`translateY(-2px)`)
- Primary card has a gradient background
- 12px border radius (friendliest in the app)
- 48px icon circles

This is the first thing a user sees after login. It looks like every AI SaaS landing page from 2024. The rest of the app is better — but first impressions matter.

**Cockpit fix:** Replace the flowing tool grid with a fixed sidebar or categorized list. Tools should be organized by workflow (Investigate → Analyze → Control → Report), not scattered in a grid. Think VS Code's activity bar or Bloomberg Terminal's panel selector — tools are always in the same position.

### 3. Hover Effects Are Inconsistent

| Template | Hover Effect |
|----------|-------------|
| Base `.card` | Border color change only |
| Dashboard `.tool-card` | Border + bg + `translateY(-2px)` + shadow |
| Hoshin `.site-card` | Border + `translateY(-2px)` + shadow |
| Loop `.loop-kpi` | Border color only |
| Investigation `.inv-card` | None |

**The problem:** Some cards lift, some glow, some do nothing. The interaction language is inconsistent.

**Cockpit fix:** One hover pattern for all interactive cards: border-color shifts to accent. That's it. No lifts, no shadows on hover. Cockpits don't bounce when you hover over a gauge.

### 4. Button Styles Diverge Per Template

- Base: ghost buttons (dim bg + colored text/border)
- Dashboard: some filled primaries
- Loop: slightly more rounded (6px)
- Hoshin: filled primaries, outlined secondaries
- Investigation: similar to base

**Cockpit fix:** Two button types only:
- **Primary action:** filled accent background, white text
- **Secondary/everything else:** ghost (transparent bg, accent border/text)

No template-specific overrides. If a button looks different in the loop dashboard vs the investigation workspace, the design system is broken.

### 5. KPI Cards Are Good But Not Uniform

Every dashboard has KPI strips, but they're all slightly different:
- Loop: 140px minmax, 12px gap, 10px radius, 26px value
- Safety: fixed 5-col, 16px gap, 10px radius, 32px value
- Hoshin: fixed 4-col, 16px gap, 10px radius, 28px value
- Internal: 140px minmax, auto gap, 6px radius

**Cockpit fix:** One `.kpi-strip` class in `base_app.html`. Fixed column count per breakpoint. Same value font size (28px JetBrains Mono). Same radius (6px). Every dashboard imports the same component.

---

## What's Actually Good (Keep These)

| Pattern | Why it works |
|---------|-------------|
| **Dark forest palette** | Distinctive, professional, easy on eyes. Not generic dark mode. |
| **Inter + JetBrains Mono** | Perfect pairing. Inter is clean without being cold. Mono for values is industry-standard. |
| **Uppercase section headers** | Creates hierarchy without being loud. Professional. |
| **Border-only cards (no shadow)** | Dense, information-focused. Shadows would add visual noise. |
| **Semantic grids** (fixed 3/4/5 col) | The GOOD grids. Intentional layout, not random flow. |
| **Analysis workbench layout** | IDE-style is correct for this tool. Sidebar + workspace + status. |
| **Investigation 3-pane** | Also correct. Sidebar + main + tool panel. |
| **Table styling** | Uppercase headers, hover rows, light borders. Professional data display. |
| **Status pills** | Semantic colors, fully rounded, small. Correct pattern. |
| **CSS variable theming** | 6 themes, all internally consistent. Rare to see this done well. |

---

## Cockpit Direction: What Changes

### Mental Model

**Before:** "Tools laid out on a wall" (grid of cards, browse and click)
**After:** "Instruments in fixed positions" (always know where everything is)

A cockpit has:
- Fixed panel positions (instruments don't rearrange)
- Dense information display (no wasted space)
- Clear visual hierarchy (primary instruments are larger/brighter)
- Consistent interaction patterns (every toggle works the same way)
- Status indicators that don't require reading (color + position convey meaning)

### Specific Widget Changes to Discuss

1. **Tool navigation:** Sidebar categories or tabbed panel instead of card grid
2. **KPI strips:** Standardized single component, denser, no rounded-card look
3. **Card radius:** 6px everywhere, no exceptions
4. **Hover effects:** Border-color only, no lifts
5. **Button standardization:** Two types, one global stylesheet section
6. **Dashboard home:** Replace card grid with activity-focused layout (recent, alerts, quick actions)

---

## Files to Touch (Estimated)

| File | Change | Risk |
|------|--------|------|
| `base_app.html` | Standardize `.card`, `.btn`, `.kpi` classes. Add `.kpi-strip`. Normalize radius to 6px. | LOW — additive CSS |
| `dashboard.html` | Restructure tool grid to categorized sidebar or panel layout | MEDIUM — visual change to home screen |
| `loop_dashboard.html` | Use standardized `.kpi-strip`, normalize radius | LOW |
| `safety_app.html` | Same KPI standardization | LOW |
| `hoshin.html` | Same + remove hover lift on site cards | LOW |
| `projects.html` | Normalize card radius from 12px to 6px | LOW |
| All playbooks (5) | Normalize TOC grid cards | LOW |

---

## What I Need From Eric

1. **Is the main dashboard the priority?** That's the biggest single change (card grid → categorized layout).
2. **How dense do you want it?** Bloomberg-dense or Grafana-dense? The current spacing is comfortable (16px gaps). Cockpit might go tighter (8-12px).
3. **Sidebar navigation — expand or replace?** The current sidebar is a simple link list. Do you want it to become the primary tool navigator (like VS Code's activity bar)?
4. **Any templates that should NOT change?** The analysis workbench is already IDE-like — it might be fine as-is.
