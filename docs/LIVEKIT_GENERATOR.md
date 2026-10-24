# LiveKit generator integration

`FloatVideoGenerator` turns live audio into synchronized LiveKit media. It is
the audio-driven counterpart to the file and in-flight stream generators: the
conversational agent pushes TTS PCM into it, it runs FLOAT in the background,
and it yields `(rtc.VideoFrame, rtc.AudioFrame)` pairs that `AvatarRunner`
publishes into a LiveKit room.

This document covers the low-latency generator integration in
`fast_float/livekit/generator.py`, its timing model, its tuning knobs, and how
the pieces fit together. For the transport, token and fallback design, see
[`LIVEKIT_INTEGRATION.md`](LIVEKIT_INTEGRATION.md).

---

## 1. Where it sits

```
agent TTS / microphone
        │  PCM (bytes, numpy or rtc.AudioFrame)
        ▼
  FloatVideoGenerator
        │  push_audio()
        ▼
  _audio_pcm (growing bytearray)
        │  enough audio, or flush()
        ▼
  ThreadPoolExecutor(1) ── run_inference_blocking()
        │  FrameDispatcher.dispatch()
        ▼
  _LiveKitDispatcherBridge        ← pairs each frame with its 40 ms PCM slice
        │  loop.call_soon_threadsafe(_push)
        ▼
  asyncio.Queue[(video, audio)]   ← bounded; production paced to playout
        │  async iteration
        ▼
  AvatarRunner → rtc.AVSynchronizer → LiveKit room
```

The generator owns no sockets and no LiveKit objects. It is a media source:
`AvatarRunner` owns the room, the tracks and the A/V pacing.

Three generator classes share the module, but only the first one is used for
live conversational audio:

| class | input | used by |
| --- | --- | --- |
| `FloatVideoGenerator` | live PCM pushed by the agent | conversational sessions |
| `LiveStreamVideoGenerator` | decoded frames tapped from an in-flight `LiveStream` | `/livekit/publish/{id}` for a live generation |
| `MediaFileVideoGenerator` | an already-rendered MP4 | `/livekit/publish/{id}` for a finished generation |

`LiveStreamVideoGenerator` and `MediaFileVideoGenerator` are unchanged by this
integration.

---

## 2. Timing model

FLOAT conditions on a **2 s audio window** and is autoregressive over chunks.
The generator cannot change that, but it can stop waiting for a full window
before it starts the first chunk:

- **First chunk:** starts as soon as `first_chunk_sec` (default **1.0 s**) of
  audio is buffered. The chunk takes up to the full 2 s window if more audio
  is already waiting. It is still zero-padded to the 2 s window expected by
  FLOAT, but only the real samples are emitted.
- **Following chunks:** wait for the full `MIN_AUDIO_SAMPLES` (2 s of audio).
- **Idle flush:** if no audio arrives for `idle_flush_sec` (default
  **0.25 s**), whatever is buffered is generated immediately. This keeps the
  tail of an utterance from sitting in the buffer forever.
- **Explicit flush:** `flush()` does the same thing deterministically, e.g. on
  an `AudioSegmentEnd` event.

The output is therefore bursty, not a constant 25 fps source. With audio
arriving incrementally:

```
audio:   [ 1.0 s ][ 2.0 s ][ 2.0 s ] ...
frames:     25        50       50
```

If the caller pushes a whole utterance at once — which is what
`ConversationalAgent.send_speech()` does — the first chunk takes the full 2 s
window and the remainder is flushed as a shorter chunk. The 1 s threshold
mainly helps live microphone/streaming input.

`AvatarRunner` buffers and paces these bursts through `rtc.AVSynchronizer`, so
the viewer still sees steady 25 fps playback as long as generation keeps up
with real time.

**Time to first frame** is approximately:

```
audio_before_first_chunk + inference_time(first chunk)
```

For incremental input that is 1 s of audio plus one GPU chunk instead of two
seconds plus one GPU chunk. For a whole-utterance push it remains 2 s.

---

## 3. Audio input

`push_audio(audio)` accepts:

| type | handling |
| --- | --- |
| `rtc.AudioFrame` | reads PCM16, downmixes multi-channel to mono, resamples to `sample_rate` |
| `bytes` / `bytearray` / `memoryview` | assumed to be **mono PCM16 at `sample_rate`**; no conversion is performed |
| `numpy.ndarray` | float arrays are clipped to `[-1, 1]` and scaled to int16; integer arrays are cast to int16 |

Call `push_audio()` from the **event-loop thread**. It extends the buffer,
tries to start a chunk, and arms the idle-flush timer.

Resampling details:

- integer sample-rate ratios (48 kHz / 24 kHz → 16 kHz) use a box filter,
  which also low-passes;
- other ratios use linear interpolation.

For raw `bytes`, the caller is responsible for matching the generator's
`sample_rate` and channel layout.

---

## 4. A/V pairing

Every emitted video frame carries its **own** 40 ms audio slice. Nothing is
read from shared state on the consumer side, so a dropped frame takes exactly
its own audio with it — A/V cannot drift by one slice per drop.

```
TARGET_FPS          = 25
SAMPLE_RATE         = 16_000
SAMPLES_PER_FRAME   = 640          # 16_000 / 25 -> 40 ms
BYTES_PER_FRAME     = 1_280        # 640 * int16
MIN_AUDIO_SAMPLES   = 32_000       # 2 s
```

`_LiveKitDispatcherBridge` derives each frame's audio position from its frame
index:

```
start_sample = (idx - first_idx) * SAMPLES_PER_FRAME
```

- a frame is emitted only while `start_sample < valid_samples`;
- the last partial frame is zero-padded with silence;
- frames that would cover only the padded tail of the 2 s window are dropped.

Because the position comes from the frame index, a frame dropped by the
dispatcher ahead of the bridge (a `gap`) does not shift every later frame's
audio. The dropped frame's 40 ms slice is skipped with it, so A/V remain
aligned instead of replaying old audio against new video.

Example: a 1.3 s chunk is padded to 2 s for inference, but emits
`ceil(1.3 / 0.04) = 33` frames, not 50. The first 32 frames carry real audio;
the 33rd carries 0.02 s of real audio and 0.02 s of silence.

### Chunked `/generate?live=true` streams

Chunked generation renders overlapping audio windows so each window has
context at its start. The MP4 merge removes that overlapping head from later
chunks. The live path must do the same, otherwise the stream repeats the
overlap and the full-request audio used by `LiveStreamVideoGenerator` falls
out of sync.

`_generate_chunked()` calls `dispatcher.begin_chunk(trim_frames)` before each
chunk, where `trim_frames = ceil(overlap_seconds * fps)` for every chunk after
the first. `LiveStreamDispatcher.submit()` drops those leading frames without
advancing the dispatcher cursor. Frame indices therefore stay continuous
across chunks, and `idx * 40 ms` continues to select the correct audio slice.

Example: 7 s of audio in three 3 s chunks with 0.5 s overlap used to publish
200 frames (175 + 25 duplicated overlap frames) and run past the end of the
audio. It now publishes 174 frames, matching the merged MP4, with no repeated
chunk head and with audio pairing aligned through the final frame.

### File and in-flight stream publishers

The other two LiveKit generators use the shared helpers in
`fast_float/utils/audio.py` and also pair audio by media time, never by decode
order:

- `LiveStreamVideoGenerator` (the `/generate?live=true` tap) slices the
  full-request WAV with `frame_sample_bounds(idx, fps)`. Both slice boundaries
  are rounded from the frame index, so a non-integer ratio does not accumulate
  drift (30 fps alternates 533/534 samples instead of losing 1/3 sample per
  frame). The dispatcher's `idx` stays continuous across chunks and
  overlap-trimmed frames do not advance it, so the same index selects both the
  picture and its sound.
- `MediaFileVideoGenerator` decodes the file's audio once, then slices it at
  each video frame's own PTS with `slice_audio_at_time()`. One-slice-per-decoded
  frame (FIFO) pairing drifts on variable-frame-rate containers and whenever
  the audio stream starts at a non-zero offset; a positive audio start is
  honoured with silence before it, and the tail is padded with silence.

The raw WebRTC transport (`fast_float/routers/webrtc.py`) uses the same helpers:
`LiveVideoTrack` publishes the index of every frame it hands to the RTP sender,
and `LiveAudioTrack` slices the request audio at exactly that index. It
replaced a `MediaPlayer`, which always started at t=0 even when the viewer
joined a render already in progress, so the soundtrack used to be seconds away
from the picture.

---

## 5. Cancellation and backpressure

### `clear_buffer()` — barge-in / interrupt

```python
generator.clear_buffer()
```

- increments the internal generation counter;
- clears buffered PCM and every queued frame;
- marks in-flight inference stale: the next `dispatch()` raises the internal
  `_Cancelled` exception. The bridge sets `abort_on_error`, so the dispatcher
  signals the pipeline's stop event and generation aborts instead of running
  to completion for audio nobody will hear;
- leaves the generator running and ready for the next utterance.

An in-flight GPU call is not killed mid-kernel; it stops at the next frame
dispatch and releases the semaphore/executor slot.

### Bounded output queue and production pacing

The output queue holds at most `max_queue_frames` pairs (default **128**,
about 5.1 s at 25 fps). Inference is much faster than real time, so without
pacing a whole utterance is queued in a burst and the oldest frames are
discarded — audible as a jump cut in the middle of a sentence.

`_pump()` therefore only starts a chunk when the queue has room for the full
burst a chunk can produce (a 2 s window = 50 frames, plus one chunk of
headroom for frames already scheduled on the loop). Production waits for the
consumer instead of dropping, so normal playback delivers every frame in
order:

```
queue:  ... 28 frames  → start chunk → ... 78 frames → drain to 28 → next
```

A queue configured smaller than one chunk cannot be paced (waiting would
deadlock), so it keeps the old drop-oldest behaviour. If the consumer still
falls more than a whole queue behind (or the loop stalls), the oldest pair is
discarded so the stream stays near live; drops are counted and logged every
25:

```
[livekit-gen] consumer behind; dropped 25 frames total
```

Because video and audio are discarded together, the surviving frames remain
in sync.

The `/generate?live=true` tap (`LiveStream`) renders much faster than real
time (100+ fps for a 25 fps stream), so a small per-subscriber queue would
overflow and skip seconds of the preview. Every real-time transport
(`LiveStreamVideoGenerator` for LiveKit, the raw WebRTC offer, and the
WebSocket stream) therefore subscribes with a queue sized from the stream's
frame dimensions by `live_queue_size()` (in `fast_float/streaming/live.py`):
a 1 GiB byte budget, clamped to 128–4096 frames. At 512×512×3 that is
~1365 frames (≈55 s), so a normal single-pass render is buffered completely
and played out in order; a multi-minute render still has a hard memory
ceiling. Only if the render outruns that budget does the queue drop the
oldest frame, which is counted and logged
(`subscriber_dropped_frames` in `metrics()`).

`AvatarRunner` paces its forward loop at `options.video_fps` **before** pushing
a frame's video and audio into `AVSynchronizer`. The synchronizer queues video
(100 ms by default) but captures audio immediately, so a bursty generator used
to let the soundtrack lead the picture by up to two frames (~80 ms). Pacing the
loop keeps the video queue near empty, so a frame's audio and video are captured
together. The same pacing applies to the file and in-flight stream publishers.

Note that the publisher's video track must also be configured at the source
frame rate: LiveKit's native default caps a track at 20 fps, which used to drop
every fifth frame of a 25 fps stream inside the encoder (before the frames
ever reached the network). See `LIVEKIT_INTEGRATION.md` §5.

### `stop()` — terminal

`stop()` invalidates the current generation, cancels the idle timer, shuts
down the owned executor, and signals end-of-stream. It deliberately **does not
flush the queue**: frames already generated are real audio and are played out
by the consumer before the iterator ends (if the queue is full, the sentinel
is not enqueued; `_stop_pending` makes the iterator exit once the queue is
drained). `AvatarRunner.stop()` bounds that drain with `DRAIN_TIMEOUT_S` before
unpublishing the tracks. A stopped generator is not restarted; create a new one.

---

## 6. Threading and GPU use

- One **dedicated** `ThreadPoolExecutor(max_workers=1)` runs inference, so GPU
  work is serialized and does not compete with the default executor used for
  DNS/file I/O elsewhere in the agent.
- The temp WAV is written **before** acquiring the optional inference
  semaphore, so file I/O does not hold the GPU slot.
- Temp WAVs go to `/dev/shm` when available (no disk I/O per chunk), otherwise
  the system temp directory. They are deleted after each chunk.
- `FrameDispatcher.dispatch()` runs on the dispatcher's dedicated delivery
  thread; results cross back to asyncio through `loop.call_soon_threadsafe`.
- Each chunk's bridge is shut down when inference returns, so the delivery
  thread does not leak for the lifetime of the session.
- The bridge disables the dispatcher's lag-drop policy (`max_lag_ms = 0`) and
  allows a full 2 s burst (`max_queue_frames = 128`). The generator's own
  bounded queue is the single backpressure point; otherwise the dispatcher
  could discard half of a 50-frame burst before delivery. `_pump()` paces
  production so the queue never overflows under normal playout.
- An optional `semaphore` lets the whole application share one GPU slot with
  the HTTP `/generate` path. Passing an `executor` transfers ownership to the
  caller: `stop()` will **not** shut it down.

---

## 7. Configuration reference

All constructor parameters are keyword-only:

| parameter | default | meaning |
| --- | --- | --- |
| `agent` | `None` | loaded FLOAT engine; `None` selects the test pattern |
| `image_path` | `None` | reference portrait; required for real inference |
| `loop` | running loop | event loop used for tasks and thread-safe pushes |
| `width`, `height` | `512`, `512` | generated frame size |
| `fps` | `25.0` | frame rate used for PTS and frame counts |
| `sample_rate` | `16000` | PCM sample rate |
| `semaphore` | `None` | optional shared inference lock |
| `first_chunk_sec` | `1.0` | audio needed before the first chunk (clamped to 2 s) |
| `idle_flush_sec` | `0.25` | flush buffered tail after this silence; `None` disables |
| `max_queue_frames` | `128` | bounded output queue length |
| `executor` | `None` | custom inference executor; caller keeps ownership |

Runtime attributes:

| attribute | meaning |
| --- | --- |
| `on_first_frame` | optional callback (sync or async) fired once, before the first frame is yielded |
| `last_chunk_rtf` | last chunk's `inference_time / audio_time`; must stay below `1.0` |
| `_dropped_frames` | diagnostics: pairs dropped because the consumer fell more than a whole queue behind |

When `last_chunk_rtf >= 1.0`, the generator logs:

```
[livekit-gen] slower than real time: RTF=1.23 (consider lower nfe / smaller chunks)
```

Sustained RTF ≥ 1 means the GPU cannot keep up with the incoming audio. Options
are fewer `nfe` steps or a smaller model/backend; otherwise the queue drains
and playback starves (silence between chunks), because production is paced to
playout rather than dropping frames.

---

## 8. Using it

### Directly

```python
import asyncio
import numpy as np

from fast_float.livekit.generator import FloatVideoGenerator

async def main() -> None:
    gen = FloatVideoGenerator(agent=engine, image_path="images/portrait.jpg")

    # Called from the event-loop thread, e.g. from a TTS callback.
    gen.push_audio(np.zeros(16_000, dtype=np.int16).tobytes())  # 1.0 s
    gen.flush()                                                 # end of utterance

    async for item in gen:
        video_frame, audio_frame = item
        # hand both to rtc.AVSynchronizer / your own publisher
        ...

asyncio.run(main())
```

In the repository, `ConversationalAgent.send_speech()` already does the
`push_audio()` + `flush()` pair for every utterance.

### With `AvatarRunner`

```python
from fast_float.livekit.worker import AvatarOptions, AvatarRunner

runner = AvatarRunner(
    room,
    generator,
    options=AvatarOptions(video_width=512, video_height=512, video_fps=25.0),
    agent_identity=agent_identity,
)
await runner.start()
# ... push audio into the generator ...
await runner.stop()
```

`AvatarRunner` publishes `avatar_video` (camera source) and `avatar_audio`
(microphone source), sets `lk.publish_on_behalf`, and sends the
`lk.playback_started` / `lk.playback_finished` RPCs.

### Inside a conversation session

`ConversationSession.start()` builds the generator with the loaded engine and
the shared render lock:

```python
self.generator = FloatVideoGenerator(
    agent=engine,
    image_path=self.image_path,
    semaphore=_render_lock,
)
```

The HTTP surface is:

| endpoint | purpose |
| --- | --- |
| `POST /livekit/conversation/start` | create the session, join rooms, publish avatar tracks |
| `POST /livekit/conversation/speak` | push one utterance's TTS audio |
| `POST /livekit/conversation/stop` | stop the generator and leave the rooms |

---

## 9. Test-pattern fallback

When `agent is None` **or** `image_path is None`, the generator skips inference
and emits a deterministic colour-ramp pattern, one frame per 40 ms of **real**
audio (the same valid-sample trimming applies). This is what the unit tests
use, and it makes it possible to exercise the LiveKit publishing path on a box
without a GPU/model:

```python
gen = FloatVideoGenerator(loop=asyncio.get_running_loop())  # test pattern
```

No temp WAV is written in this path.

---

## 10. Migration notes

The replacement keeps the public interface (`push_audio`, `flush`,
`clear_buffer`, `stop`, async iteration, `on_first_frame`) and adds optional
constructor parameters. Private internals changed:

- `_LiveKitDispatcherBridge` no longer takes `(queue, loop, audio_pcm)`. It now
  takes `(loop, push, audio_pcm, valid_samples, gen, is_current)`.
- `_LiveKitDispatcherBridge.close()` is a **no-op**. The generator owns the
  stream sentinel; a per-chunk `close()` must not end the stream. The bridge is
  still explicitly `shutdown()` after each chunk to stop its delivery thread.
- Emitted items always carry an audio slice when audio exists; the consumer
  receives `(video_frame, audio_frame)` tuples rather than video-only frames.
- `FrameDispatcher.dispatch()` can raise `_Cancelled` on stale work. The bridge
  sets `abort_on_error`, so the dispatcher's error path trips the pipeline's
  stop event. `_Cancelled` is logged at debug level, not as an error.
- Audio position is computed from the frame index, not a running sample
  counter. This keeps later frames aligned after a dispatcher-level gap and
  avoids colliding with `FrameDispatcher`'s private frame cursor.
- A dedicated inference executor replaces the default executor.

---

## 11. Verification

The targeted suite:

```bash
.venv/bin/python -m pytest \
  tests/test_livekit_tokens.py \
  tests/test_livekit_worker.py \
  tests/test_stun.py \
  tests/test_webrtc_trickle.py \
  tests/test_live_stream.py \
  tests/test_frame_dispatcher.py -q
```

Current result: **68 passed** (14 token, 17 worker/generator, 9 STUN,
3 trickle-ICE, 11 live-stream, 14 frame-dispatcher). The generator tests cover
the frame-index audio pairing, a dispatcher gap, per-chunk dispatcher shutdown,
and cancellation aborting the pipeline. The live-stream tests cover fan-out,
frame PTS, and overlap-head trimming for chunked `/generate?live=true`.

The generator was also exercised against a real LiveKit Cloud room: a
`FloatVideoGenerator` in test-pattern mode was published through `AvatarRunner`
and a second participant subscribed to both tracks, receiving video and audio
frames. For a GPU-backed check, use the conversational session on a machine
with the model loaded and watch `last_chunk_rtf` in the logs.
