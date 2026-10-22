# LiveKit integration plan

LiveKit becomes the main WebRTC backend: one transport for the live preview
and the conversational avatar, with a real SFU in front of it. FLOAT keeps
rendering the video — LiveKit moves the pixels, the rooms and the viewers.

This plan is grounded in the [LiveKit avatar docs](https://docs.livekit.io/agents/models/avatar/),
the [`AvatarRunner` reference](https://docs.livekit.io/reference/python/livekit/agents/voice/avatar/),
the [voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai/) and the
[custom avatar worker docs](https://docs.livekit.io/agents/models/avatar/#custom-avatar-workers).

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
  accumulates the incoming audio; the async iterator yields `rtc.VideoFrame`s
  (RGB24, 512×512, 25 fps) built from the frames the pipeline taps.
- It reuses today's pipeline unchanged: the audio buffer is written to a temp
  wav, `agent.run_inference(..., dispatcher=...)` runs, and the existing
  `FrameDispatcher` delivers `(idx, frames, pts_ms)` — the generator converts
  each frame and yields it with its pts.
- `AvatarRunner(room, audio_recv=DataStreamAudioReceiver(room, sender_identity=AGENT_IDENTITY), video_gen=generator, options=AvatarOptions(video_width=512, video_height=512, video_fps=25, audio_sample_rate=16000, audio_channels=1))`.
  `AvatarRunner` publishes `avatar_video` (SOURCE_CAMERA) and `avatar_audio`
  (SOURCE_MICROPHONE) and sends the `lk.playback_started` RPC for us.
- **Thread bridge**: the pipeline runs in a thread pool and the SDK is
  asyncio — the same `queue.Queue` + `loop.call_soon_threadsafe` pattern as
  `streaming/dispatcher.py::FrameQueue._push`.
- The worker takes the same `inference_lock` as the HTTP path: one GPU, one
  inference at a time.
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

## 8. Open decisions

1. **LiveKit Cloud or self-hosted?** (self-hosted fits this repo; Cloud is
   faster for a first contact.)
2. **Keep aiortc as a flag-gated fallback for one release?** (recommended yes.)
3. **Is the conversational mode in scope now?** I would do "generate → watch
   live" first: it reuses today's pipeline and needs no LLM/TTS, and it
   de-risks the whole integration before we add a conversation loop.
