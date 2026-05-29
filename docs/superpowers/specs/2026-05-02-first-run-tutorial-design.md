# First-Run Tutorial — "Aha First" Design

**Date:** 2026-05-02
**Workstream:** svend-tutorial
**Status:** Design approved, pending implementation plan

## Problem

19 of 24 free users never ran a single query. Registration lands on an empty dashboard with no guidance. The onboarding survey exists at `/app/onboarding/` but nothing sends users there. Email verification only gates AI chat, giving no reason to verify. Activation is the #1 growth bottleneck.

## Approach: "Aha First"

Get users to their first Bayesian capability analysis in under 2 minutes, before asking them to invest anything (survey, verification). Value first, commitment second.

## Flow

```
Register → auto-login → /app/tutorial/
  → Renders analysis workbench template in tutorial mode (same template, flag toggles tutorial behavior)
  → Pre-loaded Bayesian capability analysis (sample dataset)
  → Spotlight/tooltip walkthrough (7 steps, ~2 minutes)
  → Tutorial completes → abbreviated survey (1 step, inline or modal)
  → "Verify your email to run your own analyses"
  → Land in /app/
```

`/app/tutorial/` is NOT a separate template. It serves the analysis workbench template with a `tutorial=true` context variable that triggers spotlight.js initialization and pre-loads the sample dataset. This ensures the tutorial matches the real product exactly.

## 1. Tutorial Content

### Sample Dataset
- Universally relatable manufacturing scenario (e.g., shaft diameter measurements)
- ~50 data points, pre-loaded, no user upload required
- Clear LSL/USL so the capability story is obvious
- Fixture or hardcoded in the tutorial view — no database records created in the user's account

### Spotlight Sequence (7 steps)

| Step | Target | Text | User Action |
|------|--------|------|-------------|
| 1 | Data table | "We've loaded 50 diameter measurements from a turning process." | None |
| 2 | LSL/USL fields | "These are the customer requirements." Pre-filled, editable. | None |
| 3 | Run button | "Run the analysis to see your process capability." | Click Run |
| 4 | Cp/Cpk results | "Cpk = 1.32 means your process is capable, with margin." | None |
| 5 | Credible interval / posterior | "Unlike traditional Cpk, this gives you a confidence range — not just a point estimate." | None |
| 6 | Narrative section | "Svend tells you what to do with these numbers, not just what they are." | None |
| 7 | Completion card | "That was a real analysis. Run your own with your data." | CTA → survey |

### What the tutorial does NOT do
- Does not teach SPC theory — assumes quality professional audience
- Does not tour the whole app — just this one analysis flow
- Does not require any data entry beyond clicking Run

## 2. Verification Gate

### Current State
- Verification only gates AI chat endpoint (`api/views.py` ~line 183)
- No urgency to verify — everything else works without it

### New Behavior
- Tutorial run: no verification required (the freebie)
- After tutorial: prompt to verify ("Verify your email to unlock your 5 free analyses")
- Unverified users in analysis workbench: Run button disabled with verify CTA (interstitial, not hard block on page access)
- Verification email subject: "One click to start analyzing your own data"
- Once verified: full free tier access (5 runs + any additional features opened up)
- AI chat gate: unchanged (verified users only)

## 3. Spotlight/Tooltip Infrastructure

### Architecture
- `static/js/spotlight.js` — reusable engine, not a one-off
- Step definition: `{ selector, title, text, position, action? }`
- Dim overlay with cutout around spotlighted element
- Next/Back/Skip buttons on each tooltip
- Progress indicator ("Step 3 of 7")
- Tutorial definitions as JSON arrays (inline in template or small endpoint)
- CSS for overlay/tooltip in existing stylesheet

### Behavior
- "Skip tutorial" always visible — no trapping users
- If skipped: still redirect to survey + verify flow
- Step 3 pauses until Run click and results render — no fake timeouts
- Mobile: not a priority (desktop quality engineers), but tooltips reposition to bottom-sheet if needed

### Not Building
- No admin UI for tutorial authoring — JSON is fine
- No step-by-step analytics dashboard — just timestamps (24 users, not 24,000)

## 4. Backend Changes

### User Model Additions
- `tutorial_completed_at` — DateTimeField, nullable
- `tutorial_skipped_at` — DateTimeField, nullable
- Neither set = never started or bounced mid-tutorial

### Onboarding Survey Trim
- Current: 3 steps (demographics → goals/tools → confidence/urgency/challenges)
- New: 1 step post-tutorial (industry, role, primary goal only)
- Drop: org size, experience level, tools used, confidence slider, urgency slider, biggest challenge
- `onboarding_completed_at` stays as-is

### Registration Redirect
- After signup: redirect to `/app/tutorial/` instead of `/app/`
- `LOGIN_REDIRECT_URL` stays `/app/` (returning users go to dashboard)
- Redirect logic is in `register.html` JS, not Django setting

### New Endpoint
- `POST /api/auth/tutorial/complete/` — sets `tutorial_completed_at` or `tutorial_skipped_at`
- Called by spotlight.js on completion or skip

### Verification Gate in Workbench
- New check in analysis workbench view: if `not user.is_email_verified and user.tutorial_completed_at`, disable Run with verify CTA
- Tutorial route bypasses this check

### Returning Users
- Users who registered but never completed tutorial: persistent dismissable banner in `/app/` — "Take the 2-minute guided tour"

## 5. Dependencies

- New analysis workbench deployment (from analysis-workbench-migration workstream, 93% done)
- Tutorial targets the new workbench at `/app/analysis/`, not legacy DSW

## 6. Success Criteria

- New users complete the tutorial (measured by `tutorial_completed_at` rate vs. total registrations)
- Email verification rate increases (measured by `is_email_verified` rate)
- Users who complete the tutorial run at least one real analysis (measured by analysis count for users with `tutorial_completed_at`)
