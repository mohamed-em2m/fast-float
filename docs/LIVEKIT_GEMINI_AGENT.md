# Gemini Live voice agent (LiveKit Agents)

FLOAT can run its live avatar conversation with **Gemini Live** — Google's
real-time speech-to-speech model — instead of the in-process agent. A
[LiveKit Agents](https://docs.livekit.io/agents/) worker owns the
conversation, Gemini answers in its own voice, and FLOAT keeps rendering the
lip-synced avatar. This doc covers the FLOAT side; the worker itself lives in
[`agents/`](../agents/README.md) as a separate uv project.

## Why a separate process

`livekit-agents[google]` resolves to `numpy 2.x`, `pydantic 2.14` and
`websockets 16`, which would break FLOAT's pinned inference stack
(`numpy==1.24.4`, `pydantic==2.7.0`, `websockets<12`). The worker therefore
runs in its own venv and talks to FLOAT over LiveKit only:

```
Browser mic ──WebRTC──▶ LiveKit SFU ──▶ Gemini Live worker (agents/)
                        │                     │  TTS PCM over DataStream
                        │                     ▼
                        │            FLOAT avatar worker (main repo)
                        │            FLOAT renderer → avatar_video + avatar_audio
                        └─────────────────▶ Browser playback
```

## How a Gemini conversation starts

1. The Studio (or any client) calls `POST /livekit/conversation/start` with
   `"agent": "gemini"`.
2. `fast_float/livekit/session.py` connects `float-avatar` with
   `VideoGrants(agent=True)` — participant kind `AGENT`, so
   `livekit-agents`' RoomIO ignores its synthesized voice and it never loops
   back into the conversation.
3. If `FLOAT_GEMINI_AUTODISPATCH` is on (default), the API creates a LiveKit
   agent dispatch for `float-gemini` into the room via `LiveKitAPI`; the
   worker joins, connects Gemini Live, and starts listening to the room's
   microphone track.
4. Gemini's replies stream to `float-avatar` on the `lk.audio_stream`
   DataStream topic. FLOAT's `DataStreamAudioReceiver` hands the PCM to
   `FloatVideoGenerator`, which renders at 25 fps and publishes synchronized
   `avatar_video` + `avatar_audio` through `AgentAvatarRunner`.
5. The browser watches the room like any other stream; in Gemini mode it also
   publishes its microphone (token grant `publish_mic: true`).

### Wire protocol between agent and avatar

The worker sends audio with LiveKit's `DataStreamAudioOutput`; FLOAT speaks
that protocol from `fast_float/livekit/receiver.py` with no dependency on
`livekit-agents`:

| direction | message | meaning |
| --- | --- | --- |
| agent → avatar | `lk.audio_stream` segment hashes | TTS PCM for a reply segment |
| agent → avatar | `lk.playback_started` | first audio reached the room |
| agent → avatar | `lk.clear_buffer` (RPC) | barge-in: drop queued audio/video |
| avatar → agent | `lk.playback_finished` JSON (RPC) | the segment finished (or was interrupted) |
| avatar → agent | `lk.publish_on_behalf` attribute | points room UIs at the right tracks |

## Voice tools

Gemini can call FLOAT's API and control the room through
[`function_tool`s](../agents/float_agent/agent.py): `float_health`,
`float_gpu`, `generation_progress`, `conversation_status` and
`end_conversation`. The `FloatApi` client hits the FLOAT API configured by
`FLOAT_API_URL`.

## Configuration

FLOAT-side (main repo, `fast_float/livekit/settings.py`):

| env | default | meaning |
| --- | --- | --- |
| `FLOAT_AVATAR_IDENTITY` | `float-avatar` | LiveKit identity of the rendering worker |
| `FLOAT_GEMINI_AGENT_NAME` | `float-gemini` | agent name registered by the worker |
| `FLOAT_GEMINI_AUTODISPATCH` | on | auto-dispatch the agent on `agent: "gemini"` start |

Worker-side (see [`agents/.env.example`](../agents/.env.example)):
`LIVEKIT_URL` + `LIVEKIT_API_KEY` + `LIVEKIT_API_SECRET`, `GOOGLE_API_KEY`
(or Vertex AI flags), and optional `GEMINI_MODEL` / `GEMINI_VOICE` /
`GEMINI_LANGUAGE`.

## Run it

```bash
# 1. LiveKit SFU (see docker-compose.livekit.yml)
# 2. FLOAT API (this repo)
# 3. the Gemini worker
cd agents
uv sync
cp .env.example .env    # fill in keys
uv run python -m float_agent.agent dev
```

Then open the Studio → **Live audio** → engine **Gemini Live** → **Go Live**.
When the agent joins and streams its first audio, the API reports
`agent_dispatched` / `agent_connected` on
`GET /livekit/conversation/status?room=…`.

## Notes

- The worker needs **no GPU**; it negotiates Gemini Live and forwards audio.
- Keep the built-in engine as the default: `agent` defaults to `"builtin"`,
  so existing clients and tests are unchanged.
- Barge-in in Gemini mode stops the avatar's queued frames immediately via
  `lk.clear_buffer`; the agent keeps its own turn state.
- The worker uses one Gemini Live session per room job; `AgentServer` isolates
  jobs per process, so concurrent rooms scale across cores.
