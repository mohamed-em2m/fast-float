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
  asyncio.Queue[(video, audio)]   ← bounded, drops oldest on lag
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

### Bounded output queue

The output queue holds at most `max_queue_frames` pairs (default **128**,
about 5.1 s at 25 fps). When it is full, the **oldest** pair is discarded so
the stream stays near live rather than playing old frames late. Dropped pairs
are counted and logged every 25 drops:

```
[livekit-gen] consumer behind; dropped 25 frames total
```

Because video and audio are discarded together, the surviving frames remain
in sync.

### `stop()` — terminal

`stop()` invalidates the current generation, cancels the idle timer, shuts
down the owned executor, makes room in the queue, and inserts the sentinel.
The async iterator then ends and `AvatarRunner` performs its normal playout
and RPC cleanup. A stopped generator is not restarted; create a new one.

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
  could discard half of a 50-frame burst before delivery.
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
| `_dropped_frames` | diagnostics: pairs dropped because the consumer lagged |

When `last_chunk_rtf >= 1.0`, the generator logs:

```
[livekit-gen] slower than real time: RTF=1.23 (consider lower nfe / smaller chunks)
```

Sustained RTF ≥ 1 means the GPU cannot keep up with the incoming audio. Options
are fewer `nfe` steps, a smaller model/backend, or accepting a growing latency
(the bounded queue will start dropping oldest frames).

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
  tests/test_webrtc_trickle.py -q
```

Current result: **43 passed** (14 token, 17 worker/generator, 9 STUN,
3 trickle-ICE). The generator tests cover the frame-index audio pairing, a
dispatcher gap, per-chunk dispatcher shutdown, and cancellation aborting the
pipeline.

The generator was also exercised against a real LiveKit Cloud room: a
`FloatVideoGenerator` in test-pattern mode was published through `AvatarRunner`
and a second participant subscribed to both tracks, receiving video and audio
frames. For a GPU-backed check, use the conversational session on a machine
with the model loaded and watch `last_chunk_rtf` in the logs.
