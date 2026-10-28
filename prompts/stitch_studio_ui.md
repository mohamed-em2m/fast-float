# FLOAT Studio — Stitch prompt pack

Copy-paste prompts for Google Stitch (or v0 / Lovable / Figma AI) to redesign
**FLOAT Studio** — a dark, premium browser studio that turns a **portrait photo

+ speech clip** into a talking-portrait video.

## How to use Stitch with this file

Stitch omits components when a prompt is too long (>~5000 chars), and edits
work best **one screen + 1–2 changes at a time**. So:

1. Paste **Prompt 0** first (design system). It sets the tokens for everything after.
2. Generate **one screen per prompt** (Prompts 1–5), in order.
3. Use the **follow-up prompts** (§6) to iterate — one change per generation.
4. Writing/refining prompts is free; only hitting Generate costs. Draft here first.

Stack reality the designs must map onto: **React 19 + Vite, lucide-react icons,
plain CSS custom properties** (tokens in Prompt 0). Keep component boundaries:
`Header`, `PortraitCanvas`, `AudioEngine`, `ParameterControls`, `StudioStage`,
`TelemetryDashboard`. Same-origin `/api` contract — no backend changes.

---

## Prompt 0 — Design system (paste first, generate once)

> A dark premium creative-tool design system for "FLOAT Studio", a browser app
> for AI talking-portrait videos. Vibe: Figma / Runway / ElevenLabs — dense but
> calm, professional, GPU-grade. Dark glassmorphism: background #07080d, cards
> rgba(15,18,28,0.74) with 16px backdrop blur and 1px rgba(255,255,255,0.07)
> borders. Accent gradient indigo #6d7cff → cyan #2dd4e8, warm variant ending in
> rose #ff6b8a. Success #3ddc97, warning #f5b455, danger #ff5d7a. Text #f4f6fb /
> muted #9aa4b8 / faint #6b7488. Fonts: Outfit for display, Plus Jakarta Sans
> for UI, JetBrains Mono for numbers/data. Radius 7/10/14/20/26px. Shadows soft
> black plus a subtle indigo glow on primary actions. Generate a token sheet:
> color swatches, type scale, button variants (primary gradient, ghost, danger),
> badges/pills, sliders, toggles, cards, toasts, focus rings. WCAG AA contrast,
> visible focus states, prefers-reduced-motion respected.

---

## Prompt 1 — Main studio screen, desktop (the core redesign)

> A desktop studio screen for FLOAT Studio, two-column layout under a slim glass
> header. Header left: sparkle mark + "FLOAT Studio" wordmark, "v5.1 NVENC" badge,
> subtitle "Fast Flow Matching Talking Portrait Generator". Header right: status
> pill (Server Online / Model Ready / Disconnected), GPU mini-widget (allocated /
> total VRAM + usage bar + clear-cache icon), refresh and config gear buttons.
> LEFT column "Creator workbench": (1) Reference Portrait card — large preview,
> drag-and-drop upload zone, Change Photo / Clear buttons, sample thumbnail pills
> below; (2) Driving Speech card — three tabs Microphone / Upload / Sample, live
> waveform + REC indicator + timer when recording, audio player with scrub bar
> when ready, "Audio Ready" badge; (3) Generation Parameters card — emotion grid
> (Speech-to-Emotion + neutral, happy, sad, angry, surprise, disgust, fear with
> icons), sampling-steps slider 5–25 with live value, seed input + dice button,
> collapsible Advanced drawer. RIGHT column: (4) Studio Canvas — big Generate
> Portrait Video CTA, progress bar with stage label, video viewport 16:9 with
> MP4/WebRTC toggle, toolbar with stat chips (FPS, time, size) + Download / copy
> link / stream buttons; (5) Telemetry hub — KPI cards (FPS, latency, inference,
> prep time), GPU utilization bar, session history list with download links.
> Toasts bottom-right. 1440px, 13" laptop legible, generous whitespace, one
> unmistakable primary action.

---

## Prompt 2 — Reference portrait card (crop + "what the model sees")

> Redesign one card: the Reference Portrait picker for FLOAT Studio, same dark
> glass tokens as before. Large square preview with a "face focus" ring overlay
> showing the detected face box and the 1:1 crop the model will receive
> (512×512 output). Below the preview a small "model sees this" thumbnail strip.
> Controls: drag-and-drop / click-to-browse (JPG, PNG, WebP), Change Photo and
> Clear buttons, "Auto face crop (recommended)" toggle mapped to no_crop, and a
> row of quick-sample thumbnail pills. States to show: empty (placeholder +
> sample prompt), face found (green box + crop thumbnail), no face detected
> (amber warning with "use a front-facing photo, or enable no-crop" guidance).
> Side note for the design: uploads keep original resolution, backend center-crops
> ~1.6× the face box — the UI must communicate the crop instead of hiding it.

---

## Prompt 3 — Audio + parameters cards

> Two stacked cards for FLOAT Studio, same dark glass tokens. TOP card "Driving
> Speech": tab bar Microphone / Upload / Sample. Microphone tab: big record
> button, live waveform visualizer, elapsed timer, REC pill, stop control.
> Upload tab: drop zone for WAV/MP3/AAC, file row with name + duration + remove.
> Ready state: compact player (play/pause, scrub timeline, time labels),
> re-record/clear, green "Audio Ready" badge. BOTTOM card "Generation
> Parameters": emotion grid — Speech-to-Emotion default-selected + 7 emotions
> with lucide icons; sampling steps slider (5–25) with hints Ultra-fast /
> Balanced / High Fidelity and live value badge; seed row (numeric input +
> randomize dice); chunking row (off / window length / batch 1–8); collapsible
> Advanced drawer (Audio CFG 1–4, Identity CFG 0.5–3, Emotion CFG 0.5–3, NVENC
> preset Fast/Balanced/High, decode batch 0=Auto…64). Every control shows its
> value and a one-line "what this does" hint. Generate CTA disabled until audio
> exists.

---

## Prompt 4 — Stage, telemetry, header states

> The right column of FLOAT Studio, same dark glass tokens. TOP: "Studio Canvas"
> panel — primary Generate Portrait Video CTA (loading spinner state), progress
> bar with real percent + stage + completed/total chunks, or staged labels
> (Preprocess → Encode → Solve ODE → Encode video) for single-pass runs. Video
> viewport 16:9: placeholder/demo state, MP4 player state, WebRTC live state
> with green LIVE dot + "Switch to Standard MP4" escape hatch, failed state with
> auto-fallback message. Toolbar under viewport: FPS / inference-time / size /
> chunks chips, Download MP4, copy stream link, Start/Stop stream buttons.
> BOTTOM: "Telemetry & Performance" — 4 KPI cards (FPS + session average,
> total latency, model inference, prep & FFT), GPU bar (device name + VRAM
> track), session history rows (id, FPS, time, size, download). Header status
> pill variants: Disconnected (red, "cannot reach backend" + Config link),
> Server Online, Model Ready. Error toasts: 400 bad request, 429 server busy
> with retry, network failure — distinct actionable copy each.

---

## Prompt 5 — Tablet + mobile responsive

> The FLOAT Studio screen from the previous prompts, reflowed for tablet (768px)
> and phone (390px), same dark glass tokens. Single-column stack order:
> header (compact: wordmark + status dot + GPU icon + config) → Studio Canvas
> (CTA + progress + viewport first, so Generate is reachable without scrolling)
> → Reference Portrait (horizontal sample pills) → Driving Speech → Parameters
> (collapsed Advanced) → Telemetry (2-col KPI grid, history list). Sticky bottom
> Generate bar on mobile with progress %. Waveform and video scale fluidly,
> touch targets ≥44px, sliders thumb-enlarged. Show all three breakpoints and
> annotate what collapses, what sticks, and what moves where.

---

## 6. Follow-up iteration prompts (one per generation)

> On the studio screen, increase visual hierarchy: make Generate the only
> gradient element, demote secondary buttons to ghost, and add 8px more
> breathing room between cards.
>
> On the portrait card, replace the generic drop zone with a dashed face-frame
> motif and move the sample pills into a horizontal scroll row with labels.
>
> On the parameters card, group emotion + steps + seed as "Creative" and
> chunking + CFGs + encoder as "Performance", with a divider and section icons.
>
> Switch the telemetry KPI figures to JetBrains Mono, add delta-vs-session-average
> microcopy under FPS, and give the GPU bar amber/red thresholds at 80/95%.
>
> Restyle toasts: icon + title + one-line action, auto-dismiss 5s, error toasts
> sticky with a Retry button.
>
> Audit the whole screen for WCAG AA: fix any muted-text contrast failures,
> add visible focus rings to all interactive elements, label every slider with
> aria text.

---

## 7. Constraints to repeat when Stitch drifts

+ No light mode, no purple-blue default gradients — use the exact tokens.
+ No emoji icons — lucide-react only.
+ No new backend endpoints; the `/api` contract in `docs/UI_UX_BRIEF.md` §5 is fixed.
+ FLOAT research credits must stay one click away (About/credits affordance).
+ Every async action (generate, stream, upload, config test) needs a visible

> loading / success / error state — never a dead button.

## Examples

```bash
# Preview the current Studio while briefing Stitch (reference screenshots)
./scripts/run_all.sh                       # http://localhost:3000
PORT=9000 FRONTEND_PORT=3100 ./scripts/run_all.sh

# Full product spec behind these prompts
# docs/UI_UX_BRIEF.md  (§3 feature inventory, §5 data contracts, §6 states)
```
