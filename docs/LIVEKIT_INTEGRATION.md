# LiveKit integration plan

LiveKit becomes the main WebRTC backend: one transport for the live preview
and the conversational avatar, with a real SFU in front of it. FLOAT keeps
rendering the video — LiveKit moves the pixels, the rooms and the viewers.

This plan is grounded in the [LiveKit avatar docs](https://docs.livekit.io/agents/models/avatar/),
the [`AvatarRunner` reference](https://docs.livekit.io/reference/python/livekit/agents/voice/avatar/),
the [voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai/) and the
[custom avatar worker docs](https://docs.livekit.io/agents/models/avatar/#custom-avatar-workers).

For the audio-driven generator internals — chunk timing, A/V pairing,
backpressure, cancellation and tuning — see
[`LIVEKIT_GENERATOR.md`](LIVEKIT_GENERATOR.md).

## 1. The decision

**LiveKit is the transport; FLOAT is the avatar.** We implement FLOAT as a
custom avatar worker on LiveKit's `AvatarRunner`: it joins the room as a
second participant and publishes synchronised audio + video, exactly like the
hosted avatar providers do — except the renderer is our own model, running on
our own GPU, with no per-second billing and no external avatar service.

The hosted-provider plugins (AvatarTalk, D-ID, Simli, …) were studied as the
reference implementation of that shape, and are deliberately **not** part of
this plan: they render the video themselves, which would take FLOAT out of
the loop entirely.

Everything below follows from that one choice.

## 2. Target architecture

```
 Browser ──WebRTC──▶ LiveKit server (SFU) ──WebRTC──▶ FLOAT avatar worker
                        ▲                                    │  publishes
                        │                                    │  avatar_video +
                        │                                    │  avatar_audio
                        └── agent (STT/LLM/TTS) ──DataStream audio──▶ worker
```

Two different LiveKit participation modes, and they matter:

- **the avatar worker** — a *second participant* that publishes the rendered
  video. The browser must pick it, not the agent: the avatar worker is a
  participant of kind `agent` with the attribute `lk.publish_on_behalf` set to
  the agent's identity. In React, `useVoiceAssistant()` does this for us.
- **the agent** — owns the conversation (STT → LLM → TTS). Its audio output
  is *not* published to the room; it is streamed to the worker over a LiveKit
  data stream, and the worker publishes the synchronised result.

## 3. The constraint that decides the design: FLOAT is not a streaming avatar

A hosted provider renders continuously from the audio that arrives.
FLOAT cannot, and the plan has to respect that:

- FLOAT conditions on a **2 s audio window** (`wav2vec_sec=2` → 50 frames) and
  is **autoregressive over chunks**: chunk *k+1* needs the last 10 latents of
  chunk *k*.
- So the worker **cannot render a frame until it has ≥ 2 s of audio**, and
  every chunk arrives as a **burst of 50 frames**, not a steady 25 fps stream.
- Good news: the model runs at **~6x real time** (measured RTF, see
  `fast_float/core/perf_metrics.py`), so the worker can **pre-generate ahead**
  of playback and keep a FIFO of ready frames. Steady-state latency is then
  one chunk; the first frame waits for 2 s of audio plus one chunk of GPU.
- The `AvatarRunner` already owns the A/V pacing: it feeds our frames through
  `rtc.AVSynchronizer` against the audio it plays out, so we hand it finished
  frames and it decides when to publish them.

Consequence for the conversational path: "user stops speaking → avatar
starts" ≈ *time for the TTS to produce 2 s* + *one chunk of generation*
(≈ 0.5–1.5 s). That is the number we will publish in the perf matrix, in
place of today's "equals TTFF" placeholder.

## 4. What replaces what

| today | after |
| --- | --- |
| `fast_float/routers/webrtc.py` — aiortc peer connection, `/webrtc/offer` | `fast_float/routers/livekit.py` — mints a room token; the room does the media |
| `fast_float/routers/streaming.py` — `/ws/stream/{id}`, raw RGB over WebSocket | kept as a fallback (no camera, no SFU) |
| `fast_float/streaming/dispatcher.py` — `FrameDispatcher` taps decoded frames | **unchanged** — it is already the producer-side tap (drop/lag/gap accounting) and the worker subscribes to it |
| `frontend/src/services/api.js::createWebRtcSession` | `frontend/src/services/livekit.js` — `livekit-client` room + track selection |
| `frontend/src/components/LivePreview.jsx` | gains a "LiveKit" transport; the aiortc mode stays behind a flag for one release |

New files:

```
fast_float/livekit/__init__.py
fast_float/livekit/settings.py      # url, api key/secret, token TTL
fast_float/livekit/tokens.py        # JWT room grants for the browser
fast_float/livekit/generator.py     # FloatVideoGenerator: audio in -> rtc.VideoFrame out
fast_float/livekit/worker.py        # the avatar worker (AvatarRunner)
fast_float/livekit/agent.py         # the conversational agent (Phase 3)
fast_float/routers/livekit.py       # POST /livekit/token
```

`livekit` and `livekit-agents` go into an **optional extra**
(`pyproject.toml: [project.optional-dependencies] livekit = [...]`) so the
default install (and CI) stays as it is. Phase 1 needs only the RTC SDK plus
the `livekit.agents.voice.avatar` helpers; if that import drags in the whole
STT/LLM/TTS/VAD stack and fights our pinned `numpy==1.24.4` / `torch 2.7.1`,
we vendor the ~150-line `AvatarRunner` pattern (MIT) into `fast_float/livekit/`
and depend on the RTC SDK alone.

## 5. Phases

Each phase ships on its own and leaves the system working.

### Phase 0 — prove the transport (½ day)

- `livekit-server --dev` locally (or a free LiveKit Cloud project); `lk` CLI;
  API keys in `.env`.
- `POST /livekit/token` → `{url, token, room, identity}` (a room grant,
  expiring; the API secret never reaches the browser).
- A minimal page that joins the room and shows a **colour-bar** track
  published by a bare `rtc.VideoSource`.

*Exit criterion: the browser shows a LiveKit-published track, through our own
API. No model involved.*

### Phase 1 — FLOAT as a custom avatar worker (2–3 days)

- `FloatVideoGenerator` implements the provider interface: `push_audio(frame)`
  accumulates the incoming audio; the async iterator yields synchronized
  `(rtc.VideoFrame, rtc.AudioFrame)` pairs (RGB24, 512×512, 25 fps, 40 ms of
  audio per frame) built from the frames the pipeline taps.
- It reuses today's pipeline unchanged: the audio buffer is written to a temp
  wav, `agent.run_inference(..., dispatcher=...)` runs, and the existing
  `FrameDispatcher` delivers `(idx, frames, pts_ms)` — the bridge pairs each
  frame with its own PCM slice and pushes it to the asyncio queue.
- The first chunk starts after **1 s** of audio instead of 2 s; later chunks
  wait for 2 s, and an idle flush (0.25 s) drains the tail of an utterance.
  `clear_buffer()` invalidates in-flight work after a barge-in. See
  [`LIVEKIT_GENERATOR.md`](LIVEKIT_GENERATOR.md) for the full timing model.
- `AvatarRunner(room, generator, options=AvatarOptions(video_width=512, video_height=512, video_fps=25, audio_sample_rate=16000, audio_channels=1), agent_identity=AGENT_IDENTITY)`.
  Audio is not received by the runner: the agent pushes it into the generator
  with `push_audio()`. `AvatarRunner` publishes `avatar_video` (SOURCE_CAMERA)
  and `avatar_audio` (SOURCE_MICROPHONE) and sends the `lk.playback_started`
  RPC for us. The video track is published with an explicit
  `VideoEncoding(max_framerate=video_fps, max_bitrate=w*h*fps*0.15)`:
  LiveKit's native default caps a track at **20 fps**, which silently drops
  every fifth frame of a 25 fps avatar at the encoder (the preview stutters
  and appears to skip ahead). The bitrate term is clamped to 0.4–6 Mbps.
- **Thread bridge**: the pipeline runs in a dedicated single-worker executor
  and the SDK is asyncio — results cross back with
  `loop.call_soon_threadsafe` into an `asyncio.Queue` of paired A/V frames.
- The worker shares the `_render_lock` semaphore with the HTTP path: one GPU,
  one inference at a time.
- `FLOAT_LIVEKIT=1` adds "LiveKit" to the Studio's transport selector; the
  aiortc path stays as the default until this is proven.

*Exit criterion: a generation request renders into a LiveKit room and the
browser shows it, with the frames in sync with the audio.*

### Phase 2 — the browser side (1–2 days)

- `frontend/src/services/livekit.js`: connect, handle
  `RoomEvent.TrackSubscribed`, attach to the `<video>`.
- Track selection must pick the **avatar worker** (kind `agent` with
  `lk.publish_on_behalf`), not the agent participant — otherwise the browser
  renders the agent's (empty) track.
- `LivePreview.jsx` gains the transport; `StudioStage.jsx` gains the option.

### Phase 3 — the conversational agent (2–4 days)

- `fast_float/livekit/agent.py`: an `AgentServer` with STT/LLM/TTS.
- `session.output.audio = DataStreamAudioOutput(room, destination_identity="float-avatar", wait_playback_start=True)`
  — the TTS audio goes to our worker instead of the room, and the worker
  answers with `lk.playback_started` / `lk.playback_finished`.
- The worker renders with FLOAT and publishes the synchronised A/V.

### Phase 4 — measure and harden (1–2 days)

- Token auth per user, expiring grants; room cleanup on disconnect.
- `AvatarMetrics` (join latency, playback latency) →
  `fast_float/core/perf_metrics.py::RequestSample` → the perf matrix. This is
  where the "conversational latency" row stops being a placeholder.
- Reconnect handling, worker supervision (a GPU OOM kills one worker, not the
  room), and the "streams per GPU at real time" measured with *real* viewers
  instead of the synthetic sweep.

## 6. Risks

| risk | mitigation |
| --- | --- |
| The Python publisher encodes in **software** (libwebrtc VP8/VP9/H264) — our NVENC path is bypassed | 512×512@25 fps is ~1–2 ms/frame of CPU; measure it in Phase 1. If it hurts, keep NVENC and use LiveKit's **RTMP ingest** (NVENC → RTMP → LiveKit → WebRTC) instead of the SDK publisher |
| `livekit-agents` is heavy and may conflict with the pinned deps | optional extra; vendor `AvatarRunner` if needed (§4) |
| FLOAT needs 2 s of audio before the first frame | by design (§3); measure and publish the real number |
| One GPU, serialised inference | the worker takes `inference_lock`; one room per worker, N rooms per GPU |
| Two WebRTC stacks in the repo | the aiortc path is flag-gated for one release, then removed |
| Cloud vs self-hosted | dev: `livekit-server --dev`; prod: self-hosted (this repo is already self-hosted Docker) — `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` in the env |

## 7. What this does *not* fix

- **The cadence.** The pipeline still produces bursts of 50 frames; LiveKit
  moves the pixels, it does not make the model streaming. The 2 s audio
  window and the chunk autoregression are ours to live with (or to attack
  separately: a shorter window, or a streaming variant of the FMT).
- **The GPU.** One GPU, one inference at a time. LiveKit's SFU makes the
  *distribution* cheap; the *generation* stays the bottleneck.
- **A/V sync in the file path.** The `/generate` → MP4 path is unaffected and
  keeps its own muxing.

## 8. Decisions taken

| decision | answer |
| --- | --- |
| LiveKit deployment | **self-hosted** (`docker-compose.livekit.yml`, the LiveKit server container; a TLS domain and `wss://` in production) |
| aiortc | **kept as a flag-gated fallback** for one release, then removed |
| scope of the first milestone | the **live mode** (generate → watch live). The conversational agent (STT/LLM/TTS) comes after |
| avatar | FLOAT, always. No hosted avatar service |

## 9. The dependency finding that shapes the worker

`livekit` (the RTC SDK) requires **`numpy>=1.26`**; this repository pins
`numpy==1.24.4` because `onnx<1.17` needs numpy 1.x. They cannot coexist in
one environment:

```
uv pip install --dry-run "livekit>=1.1"
 - numpy==1.24.4
 + numpy==2.2.6      # <- would break the onnx/onnxruntime stack
```

So the split is:

- the **API process** stays as it is (numpy 1.24.4, onnx<1.17) and only needs
  `livekit-api` — the *token* API, which is numpy-free and installs cleanly
  (it is an optional dependency group: `uv sync --group livekit`);
- the **avatar worker**, if it needs the RTC SDK, gets its **own uv project**
  (its own lockfile, numpy 2 + torch + onnxruntime) — or avoids the SDK
  altogether by publishing over RTMP (below).

## 10. Phase 0 — done: the server side of the transport

Implemented and verified:

- `fast_float/livekit/settings.py` — the env-driven config (`LIVEKIT_URL`,
  `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_TOKEN_TTL_S`,
  `LIVEKIT_ROOM_PREFIX`).
- `fast_float/livekit/tokens.py` — mints the room grant with the official
  `livekit-api` (`AccessToken` + `VideoGrants`). The API secret never leaves
  the server; a viewer token has `can_publish=False` by default.
- `fast_float/routers/livekit.py` — `GET /livekit/status` (is it configured?),
  `POST /livekit/token` (mint a room-scoped token) and
  `GET /livekit/publish/{id}` (is the background publisher still running, and
  its last error, so the client can explain a fallback).
- `docker-compose.livekit.yml` — the self-hosted server.
- `tests/test_livekit_tokens.py` — 10 tests: the settings, the room-name
  validation, the claim shape, the publish flag, and the endpoint's
  503/400/200 paths.

**Verified end to end** against a real self-hosted server: a token minted by
`fast_float.livekit.tokens.issue_token` was accepted by the LiveKit
container, and the official RTC SDK joined the room as that identity and
disconnected cleanly. So the claim format is right, not just plausible.

To try it:

```bash
docker compose -f docker-compose.livekit.yml up -d
export LIVEKIT_URL=ws://127.0.0.1:7880
export LIVEKIT_API_KEY=devkey
export LIVEKIT_API_SECRET=devsecretdevsecretdevsecretdevsecret
uv sync --group livekit          # livekit-api (numpy-free)
curl -s localhost:8000/livekit/status
curl -s -X POST localhost:8000/livekit/token -H 'content-type: application/json' \
     -d '{"room": "float-demo"}'
```

### ICE without a LiveKit/coturn dependency

The API process runs a small RFC 5389 **STUN binding responder** on UDP 3478
(`fast_float/utils/stun.py`, started in the FastAPI lifespan). Both sides of
the raw WebRTC transport use it:

- the browser gets `stun_urls` from `GET /livekit/status` and passes them to
  the raw WebRTC `RTCPeerConnection` (the LiveKit client keeps the ICE/TURN
  list its own SFU returns);
- aiortc builds its `RTCConfiguration` from the same list, so it gathers a
  public reflexive candidate instead of host candidates only.

The advertised host is the one the browser used for the API (or an explicit
`[webrtc].stun_host`); a loopback host is rewritten client-side to the API
host. Configure `[webrtc].stun_enabled` / `stun_port`, or point `stun_urls`
at an external coturn — an explicit list wins and disables the built-in
responder. `run_all.sh` exports `WEBRTC_STUN_URLS` when it starts the
compose `stun` profile.

When LiveKit cannot be reached the Studio now falls back to this raw WebRTC
transport first (main-server signalling + STUN); the WebSocket transport is
only the last resort.

All three transports consume the same `LiveStream` fan-out and buffer a fast
render with `live_queue_size()` (1 GiB / 128–4096 frames, see
`fast_float/streaming/live.py`). The raw WebRTC offer previously kept only 8
frames, so a 15 s render froze the preview after about one second. When the
render finishes, the offer closes its peer connection and the viewer hands
over to the finished MP4 instead of freezing on the last frame.

STUN only discovers addresses; it does not relay media. If the browser reaches
the API through a **TCP-only edge proxy** (RunPod/Cloudflare tunnels, many
corporate NATs), WebRTC cannot open a UDP path and will never reach
`connected`. Configure a TURN relay in `[webrtc]` for those deployments:

```toml
turn_urls = ["turn:turn.example.com:3478?transport=tcp"]
# Static credential:
turn_username = "..."
turn_credential = "..."
# Or, for coturn `use-auth-secret`, put only this in secerts.toml:
# turn_secret = "..."
```

`GET /livekit/status` returns the complete `ice_servers` list (STUN + TURN,
with short-lived coturn REST credentials when `turn_secret` is set); the same
list feeds aiortc's `RTCConfiguration`. `POST /offer` returns a `peer_id`, and
browsers trickle late candidates to `POST /offer/candidate`, so a slow TURN/TCP
allocation is not lost when the initial SDP was already sent. The Studio gives
ICE 12 seconds to connect and then falls back to WebSocket with an explanatory
toast, so a TCP-only deployment no longer hangs on "Connecting…".

## 11. The transport decision still to settle: SDK publisher vs RTMP ingest

The worker has to get H.264 (or raw frames) into the room, and there are two
ways:

| | SDK publisher (`AvatarRunner`) | RTMP ingest |
| --- | --- | --- |
| encode | libwebrtc **software** (VP8/VP9) — our NVENC path is bypassed | **NVENC**, unchanged: the pipeline's H.264 goes straight out |
| dependencies | needs the RTC SDK → numpy 2 → a **separate uv project** for the worker | the **Ingress service** (a separate container) + Redis |
| latency | lowest (~100 ms) | adds the ingest buffer (~1 s) |
| A/V | the worker publishes synced audio + video | whatever we mux into the file |
| fit with the avatar-worker model | exact | we are the *publisher*, not a worker |

Measured on the self-hosted server (the container's listeners, not the docs):
the core server has **no RTMP listener** — only 7880 (http/ws) and 7881
(rtc). `LIVEKIT_RTMP_ENABLED` / `LIVEKIT_RTMP_PORT` do not exist in this
version; the only RTMP-related flag is `--ingress.rtmp_base_url`, i.e. RTMP
ingest belongs to the separate **Ingress service**, which also needs Redis.
So the RTMP route is two more containers, not a flag.

That splits the two modes cleanly:

- **the live mode** (generate → watch) can publish the NVENC'd H.264 through
  the **Ingress service** and needs neither the RTC SDK nor a second
  environment — at the price of the ingress buffer;
- **the conversational mode** (Phase 3) needs the SDK publisher, because the
  worker has to react to audio that arrives live — and therefore the separate
  numpy-2 project.

So the live mode can ship without touching the numpy pin at all, and the
second environment is only needed when the conversation loop starts.

## 12. Implementation status: Phases 0–3 completed

The plan is fully implemented and tested across backend and frontend:

- **Phase 0 (Transport foundation)**:
  - `fast_float/livekit/settings.py`: Environment-driven configuration.
  - `fast_float/livekit/tokens.py`: Room-scoped viewer JWT tokens via `livekit-api`.
  - `fast_float/routers/livekit.py`: `GET /livekit/status` and `POST /livekit/token`.
  - `docker-compose.livekit.yml`: Containerized self-hosted LiveKit SFU server.

- **Phase 1 (Avatar Worker & Video Generator)**:
  - `fast_float/livekit/generator.py`: `FloatVideoGenerator` with a 1 s first chunk, 2 s steady-state chunks, a 0.25 s idle tail flush, per-frame 40 ms audio pairing, cancellable generations, a bounded output queue and a dedicated inference executor. Also `LiveStreamVideoGenerator` for direct in-flight stream forwarding. See [`LIVEKIT_GENERATOR.md`](LIVEKIT_GENERATOR.md).
  - `fast_float/livekit/worker.py`: `AvatarOptions`, `AvatarRunner` with `rtc.AVSynchronizer` (coordinating `avatar_video` and `avatar_audio` tracks, setting `lk.publish_on_behalf` attribute), and `LiveKitLiveStreamPublisher` with background task spawning.
  - `fast_float/routers/generation.py`: Automatically launches LiveKit publisher task when `live=True` and LiveKit is configured.
  - `fast_float/streaming/live.py`: `LiveStreamDispatcher.begin_chunk()` trims the overlapping head of later audio chunks before publishing, so chunked live streams stay continuous and frame-index audio pairing does not fall behind the full-request audio.

- **Phase 2 (Browser UI)**:
  - `frontend/src/services/livekit.js`: `livekit-client` room connection, token fetching, track subscription, and attaching **both** the video and audio tracks to the same `<video>` element so they share one `MediaStream` playback clock (a separate hidden `<audio>` is only a fallback when there is no video element).
  - `frontend/src/components/LivePreview.jsx`: Added `livekit` transport rendering live tracks with connection status badges.
  - `frontend/src/components/StudioStage.jsx`: Added `LiveKit` to the `TRANSPORTS` selector.
  - `frontend/package.json`: Added `livekit-client` dependency; verified via `npm run build`.

- **Phase 3 (Conversational Agent)**:
  - `fast_float/livekit/agent.py`: `ConversationalAgent` managing conversation turns, streaming audio to worker, and coordinating via `lk.playback_started` and `lk.playback_finished` RPCs.

- **Automated Verification**:
  - `tests/test_livekit_tokens.py`: 14 tests covering settings, grants and endpoints.
  - `tests/test_livekit_worker.py`: 17 tests covering the generator, worker, synchronizer, agent, media-file playback and publish endpoints.
  - `tests/test_stun.py`: 9 tests for the built-in STUN responder and static/secret-based TURN credentials.
  - `tests/test_webrtc_trickle.py`: 3 tests for trickle ICE and graceful degradation.
  - `tests/test_live_stream.py`: 11 tests for live fan-out, PTS, and overlap-head trimming in chunked live generation.
  - `tests/test_frame_dispatcher.py`: 14 tests for ordered delivery, gaps, and backpressure policy.
  - All 68 tests passing cleanly.
