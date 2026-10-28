# FLOAT Studio — UI/UX Design Brief & Prompt

A ready-to-hand brief for a UI/UX builder (human designer or AI design tool such as
v0 / Lovable / Figma AI / Claude Artifacts). It describes the product, every feature,
the current interface, the data the UI must handle, and the design direction.

---

## 0. Copy-paste prompt

> You are a senior product designer building a high-fidelity UI for **FLOAT Studio**, a
> browser-based creative studio for generating **audio-driven talking-portrait videos**
> from a single photo plus a speech clip. The app is a dark, premium "creative tool" —
> think Figma / Runway / ElevenLabs: dense but calm, professional, trustworthy, GPU-grade.
>
> Produce a modern, polished, accessible design that covers the full workflow below.
> The stack is **React 19 + Vite + lucide-react** (no component library yet) with plain
> CSS custom properties — so deliver clean, tokenized CSS/JSX or a component spec that
> maps onto that stack. Keep the existing design language (dark glassmorphism with an
> indigo→cyan→rose accent system) but elevate it: sharper hierarchy, better empty/loading/
> error states, responsive layout, and accessible contrast and focus states.
>
> The complete product description, feature inventory, screens, data contracts, states,
> and acceptance criteria follow.

---

## 1. Product in one paragraph

FLOAT Studio turns a **portrait photo + a speech recording** into a **temporally
consistent talking-portrait video** whose lip sync and facial expression follow the audio
(and, optionally, a chosen emotion). It is a thin, fast studio over the FLOAT research
model (ICCV 2025), accelerated with NVENC encoding, batched GPU decoding, and a
TensorRT/CUDA inference path. Users compose in a single screen — pick a portrait, provide
speech (record / upload / sample), tune a few creative parameters, hit **Generate**, then
preview, live-stream over WebRTC, and download the MP4 — while live GPU telemetry and
per-request performance stats stay visible.

**Audience:** AI/creative practitioners, video editors, researchers, and demo operators who
value speed and control. They are technically comfortable but want a beautiful, low-friction
tool.

---

## 2. Workflow / journey to design

```
Choose portrait ─┐
                 ├─▶ Provide audio ─▶ Tune params (optional) ─▶ Generate ─▶ Preview ─▶ Export
Choose emotion ──┘                                                      │
                                                          (live progress + telemetry)
```

Fast path: everything defaults are sensible — a user can hit **Generate** immediately with
a sample portrait + a sample clip. Power path: the same screen exposes emotion, sampling
steps, seeds, chunking, batch sizes, guidance scales, and encoder quality.

---

## 3. Feature inventory (what the UI must expose)

### 3.1 Reference portrait

- Drag-and-drop / click-to-browse upload (PNG, JPG, WebP). Default portrait preloaded.
- Large preview with a "face focus" affordance; **Change Photo** and **Clear**.
- **Auto face crop & centering** toggle (recommended for non-aligned photos) — maps to `no_crop`.
- **Quick samples** (thumbnail pills) to load a demo portrait instantly.

### 3.2 Driving speech audio (three input modes)

- **Microphone**: record with live waveform visualizer, elapsed timer, REC indicator, stop.
- **Upload**: click or drop a file (WAV, MP3, AAC, PCM; 16 kHz recommended).
- **Sample speech**: one-click demo clip.
- **Preview player** once audio exists: play/pause, scrub timeline, time labels, re-record/clear.
- "Audio Ready" badge; format-aware (the real container is sent, backend transcodes).

### 3.3 Emotion

- Mode grid: **Speech-to-Emotion (S2E)** plus 7 directed emotions —
  `neutral, happy, sad, angry, surprise, disgust, fear` — with icons.

### 3.4 Generation parameters

- **Sampling steps (NFE)** slider, 5–25, with helper text (⚡ Ultra-fast / Balanced /
  High Fidelity).
- **Seed** numeric input + randomize.
- **Audio chunking**: off / overlapping-window length (0.1–60 s), plus **chunk batch size**
  (1–8) — for long speech, batched and progress-tracked.
- **Advanced drawer**: Audio CFG (1–4), Reference-identity CFG (0.5–3), Emotion CFG (0.5–3),
  **NVENC quality preset** (Fast / Balanced / High), **Decode batch size** (0 = Auto … 64,
  VRAM auto-tuned; manual override warns about VRAM).
- Each control shows a live value badge and a one-line "what this does" hint.

### 3.5 Generate & progress

- Primary **Generate Portrait Video** CTA (disabled until audio exists; spinner while running).
- Progress with two sources:
  - **Real** server progress when chunked (`completed_chunks / total_chunks`, percent, stage, message), polled each second.
  - **Cosmetic stage cycler** fallback for single-pass runs
    (Preprocess → Encode motion latent → Solve ODE → NVENC compress).

### 3.6 Output & playback

- Video viewport: standard **MP4** player (streamed via HTTP range) and an optional
  **WebRTC live** player (peer connection states: negotiating / connected / failed → fallback to MP4).
- Result stat chips: **FPS**, **inference time**, **file size**, and chunk count.
- Actions: **Download MP4**, **copy stream link**, **WebRTC stream / close**.
- WebRTC status overlay with a live dot and a "Switch to Standard MP4" escape hatch.

### 3.7 Connection & backend config

- Status pill: **Disconnected / Server Online / Model Ready** (tolerant of transient blips).
- **GPU widget** in the header: allocated / total VRAM, usage bar, clear-cache button.
- **Config popover**: shows the active API endpoint (default same-origin `/api` proxy vs a
  custom override), lets the user enter a URL, a loopback warning for remote browsers, and
  **Test & Save / Reset / Cancel**. Validates before committing.

### 3.8 Telemetry & history

- **KPI cards**: Inference speed (FPS, with session average), Total latency, Model inference
  time, Prep & audio FFT time.
- **GPU diagnostics bar**: device name + VRAM utilization track.
- **Session history**: recent generations with id, FPS, time, size, and download link.

### 3.9 Notifications

- Toast system (success / error / info), auto-dismissing.

---

## 4. Current interface (as-is, for reference)

Two-column "studio" grid under a glass header. Dark glassmorphism, gradients, monospace
numerals for data.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ HEADER  ✦ FLOAT Studio (v5.1 NVENC)   [status pill] [GPU widget] [⟳] [Config] │
├───────────────────────────────────┬──────────────────────────────────────────┤
│ LEFT: Creator workbench           │ RIGHT: Stage & telemetry                  │
│  1. Reference Portrait            │  Studio Canvas & Output                   │
│  2. Driving Speech Audio          │   • Generate CTA + progress               │
│  3. Generation Parameters         │   • Video viewport (MP4 / WebRTC)         │
│     (+ Advanced drawer)           │   • Toolbar: stats, stream, copy, download│
│                                   │  Telemetry & Performance Hub              │
│                                   │   • KPI grid, GPU bar, history            │
└───────────────────────────────────┴──────────────────────────────────────────┘
```

**Components:** `Header`, `PortraitCanvas`, `AudioEngine`, `ParameterControls`,
`StudioStage`, `TelemetryDashboard`. **Brand:** "FLOAT **Studio**", subtitle "Fast Flow
Matching Talking Portrait Generator", badge `v5.1 NVENC`.

**Design tokens (current):**

| Token | Value | Use |
|---|---|---|
| `--bg-dark` | `#080a10` | page background |
| `--bg-card-solid` | `#121624` | card base under glass |
| `--primary` / `--primary-light` | `#6366f1` / `#818cf8` | primary actions, indigo |
| `--cyan` | `#06b6d4` | accents, data |
| `--success` | `#10b981` | ready / online |
| `--warning` | `#f59e0b` | cautions |
| `--danger` | `#f43f5e` | errors, recording |
| `--text-main` / `--text-muted` / `--text-faint` | `#f8fafc` / `#94a3b8` / `#64748b` | text hierarchy |
| glass | `backdrop-filter: blur(16px)` | panels |

Accent gradient runs **indigo → cyan → rose** (used in the wordmark, waveform, KPI figures).

---

## 5. Data contracts the UI binds to

```jsonc
// POST /api/generate  (request)
{
  "audio_base64": "...", "audio_format": "wav|mp3|webm|ogg|mp4",
  "sample_rate": 16000, "channels": 1, "sample_width": 2,
  "image_base64": "...", "image_format": "jpg",
  "emotion": "S2E|neutral|happy|sad|angry|surprise|disgust|fear",
  "nfe": 10, "seed": 25,
  "a_cfg_scale": 2.0, "r_cfg_scale": 1.0, "e_cfg_scale": 1.0,
  "no_crop": false, "compress": true, "quality": "fast|balanced|high",
  "chunk_seconds": 0, "chunk_overlap_seconds": 0.5,
  "decode_batch_size": 0, "chunk_batch_size": 1,
  "client_request_id": "<uuid>"
}

// POST /api/generate  (response)
{ "request_id": "...", "stream_url": "...", "download_url": "...",
  "download_backup": "...", "chunks": 1, "file_size_mb": 3.2,
  "performance": { "preprocess_time": 0.4, "inference_time": 6.1,
                   "total_time": 7.2, "fps": 42.0 } }

// GET /api/progress/{id}   { status, stage, percent, completed_chunks, total_chunks, message }
// GET /api/health          { model_loaded, performance_stats: { average_fps } }
// GET /api/gpu             { info: { name }, memory: { allocated_gb, total_gb, usage_percent } }
// POST /api/gpu/clear      -> ok
// POST /api/offer          WebRTC SDP exchange  -> answer
// GET /api/stream/{id}     byte-range MP4      | GET /api/download/{id}
```

All calls go through the same-origin `/api` proxy; a user-configurable base URL overrides it.

---

## 6. States to design (must-have)

- **Empty / first-run:** default portrait + sample clip, Generate enabled, viewport shows a placeholder/demo.
- **Recording:** live waveform, REC pill + timer, pulsing stop control.
- **Audio ready:** preview player, badge, clear/re-record.
- **Generating:** progress bar (real % or staged), disabled CTA, non-blocking telemetry.
- **WebRTC:** negotiating → connected (live) → failed → auto-fallback to MP4.
- **Disconnected:** status pill offline, generator blocked with a clear "cannot reach backend" message + link to Config.
- **Errors:** 400 bad request, 429 server busy (queue full), network failure — each with distinct, actionable copy.
- **GPU constraints:** low VRAM / no CUDA — guide toward smaller batch sizes / lower decode batch.
- **Long audio:** chunked mode with per-chunk progress and chunk count in results.

---

## 7. Design direction & constraints

- **Vibe:** premium creative tooling — dark, glassy, precise; data-forward with monospace
  numerals; motion used sparingly (progress, live dots, waveform).
- **Layout:** two-column desktop studio → single column stacked (workbench → stage →
  telemetry) on tablet/mobile. Sticky header; generate CTA always reachable.
- **Hierarchy:** one unmistakable primary action (Generate); inputs clearly grouped into
  Portrait / Audio / Parameters; output and telemetry visually distinct from inputs.
- **Tokens:** keep the existing CSS custom properties; add spacing/radius/elevation scales,
  focus rings, and a motion scale if missing.
- **Accessibility:** WCAG AA contrast, visible focus states, keyboard-operable controls,
  labeled sliders, `prefers-reduced-motion` support, ARIA for the progress bar and status pill.
- **Responsiveness:** usable down to mobile; waveform and video scale fluidly.
- **Tech constraints:** React 19 + Vite, `lucide-react` icons, plain CSS (or introduce a
  tokenized styling approach) — keep the existing component boundaries and the `/api` contract;
  do not require backend changes.
- **Attribution:** FLOAT credits (see repo `README.md`) must remain accessible in the UI
  (e.g., a small "About / credits" affordance).

---

## 8. Deliverables & acceptance criteria

1. A responsive, tokenized **design system** (`:root` variables) covering color, type,
   spacing, radius, elevation, and motion.
2. Screen designs for **desktop, tablet, mobile** of the full studio, plus the **Config**
   and **Advanced** surfaces.
3. A component inventory with all states (empty, loading, recording, generating, error,
   offline, WebRTC).
4. A short **interaction spec** (progress model, WebRTC fallback, toast usage, validation).
5. Acceptance: a first-time user can go photo → audio → Generate → download without
   instructions; every async action has a visible state; the interface stays legible and
   responsive on a 13" laptop and a phone.

## Examples

```bash
# Backend + Studio (recommended for development)
./scripts/run_all.sh                       # http://localhost:3000

# Custom ports - the Vite proxy follows PORT
PORT=9000 FRONTEND_PORT=3100 ./scripts/run_all.sh

# Studio only, pointed at a remote or tunnelled backend
VITE_API_PROXY_TARGET=https://my-host.example ./scripts/run_frontend.sh

# Production build of the same UI
cd frontend && pnpm install --frozen-lockfile && pnpm build && pnpm preview
```

Browser code calls `/api/*`; the dev/preview proxy rewrites it to `/*` and
forwards to `http://localhost:${PORT}`. Transport ladder: LiveKit SFU -> raw
WebRTC (`POST /offer`, `"live": true`) -> WebSocket (`/ws/stream/{id}`) -> MP4.

Run modes and API calls: [USAGE.md](USAGE.md). Streaming protocol:
[STREAMING_PIPELINE.md](STREAMING_PIPELINE.md).
