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

The Gemini API key and model defaults live in the repo's shared config, so
one place serves both the API and the worker:

| shared place | keys |
| --- | --- |
| `fast_float/config/secerts.toml` `[gemini].api_key` | the Gemini API key (git-ignored) |
| `fast_float/config/settings.toml` `[gemini]` | `model`, `voice`, `language`, `vertexai`… |

The Studio can also supply **per-session overrides** from its header
**Config** panel (stored in the browser only): a Gemini API key and/or a
Live model name. They travel with the `/livekit/conversation/start` request
and are forwarded to the worker inside the dispatch `metadata`
(`{"gemini_api_key": "...", "gemini_model": "..."}`); the worker prefers
them over its own config for that session and never logs the key. This is a
convenience for quick sessions — for persistent values keep using
`secerts.toml` / `settings.toml` or the `GOOGLE_API_KEY` / `GEMINI_MODEL`
env vars.

Worker-side (see [`agents/.env.example`](../agents/.env.example)):
`LIVEKIT_URL` + `LIVEKIT_API_KEY` + `LIVEKIT_API_SECRET`, `GOOGLE_API_KEY`
(or Vertex AI flags), and optional `GEMINI_MODEL` / `GEMINI_VOICE` /
`GEMINI_LANGUAGE`. The shared FLOAT config is the **single source of truth**:
both processes read the same `secerts.toml` / `settings.toml` sections and a
file value wins over the env; a disagreeing env copy is ignored and logged
as stale. Only set these in `agents/.env` when the box has no shared config
— a copy here that drifts is exactly why a worker can register on a
different LiveKit project than the API dispatches into.

## Run it

```bash
# 1. LiveKit SFU (see docker-compose.livekit.yml)
# 2. FLOAT API (this repo)
# 3. the Gemini worker
cd agents
uv sync
cp .env.example .env    # optional — the shared secerts.toml/settings.toml
                        # is preferred; agents/.env is only the fallback
uv run python -m float_agent.agent start
```

`start` replaces the deprecated `dev` command (livekit-agents removed
in-process auto-reload, so `dev` only warns). A healthy worker logs

```
worker starting: agent_name=float-gemini url=wss://… config_dir=…/fast_float/config livekit_credentials=True gemini_key=True model=…
registered worker: agent_name=float-gemini url=wss://…
```

The first line is this repo's summary of what the process resolved (SFU,
agent name, config directory, credential/key presence); the second comes
from livekit-agents and proves the SFU accepted the registration. If the
process instead exits with `ws_url is required, or set LIVEKIT_URL
environment variable`, the `[livekit]` block did not resolve — check
`secerts.toml` and the worker's `FLOAT_CONFIG_DIR`. Both the worker and the
API export the resolved LiveKit values to their own environment, because
`livekit-agents` and google-genai read `os.environ` directly.

Then open the Studio → **Live audio** → engine **Gemini Live** → **Go Live**.
When the agent joins and streams its first audio, the API reports
`agent_dispatched` / `agent_connected` on
`GET /livekit/conversation/status?room=…`.

## How the agent works inside FLOAT (end to end)

A Gemini session involves four parties: the browser (mic + preview), the
LiveKit SFU, the Gemini worker (`agents/`) and the FLOAT avatar worker (this
repo). The sequence below is what happens between *Go Live* and the first
frame on screen:

```
Studio → POST /livekit/conversation/start?agent=gemini
   │
   ├─① API connects FLOAT avatar ('float-avatar', kind=AGENT) into the room
   ├─② API creates an SFU AgentDispatch for 'float-gemini' (+ Studio
   │     overrides in dispatch metadata), stores its dispatch id
   ├─③ SFU offers the job to the worker registered under 'float-gemini'
   ├─④ Worker → AgentSession(Gemini Live) → RoomIO mic track
   ├─⑤ Gemini TTS PCM streams on lk.audio_stream → DataStreamAudioReceiver
   ├─⑥ AgentAvatarRenderer renders lip-synced frames → avatar_video/audio
   └─⑦ Browser plays the avatar tracks; playback_finished RPCs close each
        segment loop
```

### Who decides which key/model is used

The value chain is layered so that *deployment* truth and *session* intent
have separate owners. Both processes (API and worker) resolve the deployment
layers identically:

| layer | wins over | belong to |
| --- | --- | --- |
| Studio dispatch metadata | everything | this session only (`gemini_api_key`, `gemini_model`) |
| `secerts.toml` / `settings.toml` `[livekit]` `[gemini]` | env | deployment: both processes must agree |
| `GOOGLE_API_KEY`, `LIVEKIT_*` env / `agents/.env` | defaults | fallback for boxes without the shared config |

Rules that keep the stack honest:

- **Studio overrides are session-scoped.** They ride in dispatch metadata
  and reach `RealtimeModel` as explicit arguments; the worker never writes
  them into `os.environ` (a process-global map that would leak a Studio key
  into later jobs). In Vertex AI mode a session key has no effect and this
  is logged.
- **The env copy never silently overrides the file.** An env value that
  differs from `secerts.toml`/`settings.toml` is ignored on both sides with
  a warning naming the variable — a drifted copy is exactly how a worker
  ends up registered on a different LiveKit project than the API
  dispatches into.
- **Studio model names are validated at the boundary.** `RealtimeModel`
  accepts any string and only fails on the Gemini websocket seconds later;
  the worker normalizes case/quotes and warns about unknown-but-plausible
  ids instead of starting a session that cannot connect.

### Dispatch lifecycle (and why "NO job" appears)

LiveKit explicit dispatch is one queued entry per room, consumed by the
first worker registered under that agent name. Per session the API:

1. creates the dispatch and remembers its id;
2. waits up to `FLOAT_GEMINI_AGENT_WAIT_S` (default 45) for an agent-kind
   participant to appear;
3. on teardown (normal stop, **or** the start timing out) deletes the same
   dispatch, so failed starts do not accumulate queue entries.

Diagnosing *“the Gemini worker did not join … NO job”*: the message lists
one line per existing dispatch. One dispatch line ⇒ the current attempt;
several lines from an old deploy ⇒ dispatches leaked by stops before this
cleanup (they can be removed via `livekit-cli dispatch delete` or by
restarting the SFU). It always means **no worker is currently registered
under the agent name on that SFU** — check in this order:

1. Is the worker up? `worker starting: …` proves it resolved config and
   `registered worker` proves the SFU accepted it. An exit with
   `ws_url is required, or set LIVEKIT_URL environment variable` means the
   resolved values never reached livekit-agents (fixed here by exporting
   them, but a hand-rolled entrypoint can still hit it).
2. Do API and worker read the **same** `fast_float/config` directory
   (`FLOAT_CONFIG_DIR`)? With file-first resolution a second checkout
   reading another directory is a silent different-deployment bug — compare
   the `config_dir=` field on `worker starting` with the API's
   `GET /livekit/status`.
3. Any stale env copies? Both processes log them as
   `"The shared file wins — remove or align the env copy"`.

### API endpoints involved

| route | role in a Gemini session |
| --- | --- |
| `POST /livekit/conversation/start` | token + room + avatar join + dispatch create |
| `GET /livekit/conversation/status` | `agent_dispatched`, `agent_connected` for the UI |
| `POST /livekit/conversation/stop` | disconnect, drain, delete the dispatch |
| `GET /livekit/status` (`?verify=1`) | readiness checks the Studio shows before going live |

## Notes

- `GET /livekit/status` returns a `checks`/`warnings` readiness list (LiveKit
  credentials, gemini key, agent name, auto-dispatch) the Studio shows in the
  Config panel and Live Audio tab. `GET /livekit/status?verify=1` probes the
  SFU for real (mint a token + dial it) so the operator can prove the
  credentials work before a session starts; the API also runs this probe once
  in the background at startup.

- The worker needs **no GPU**; it negotiates Gemini Live and forwards audio.
- Keep the built-in engine as the default: `agent` defaults to `"builtin"`,
  so existing clients and tests are unchanged.
- Barge-in in Gemini mode stops the avatar's queued frames immediately via
  `lk.clear_buffer`; the agent keeps its own turn state.
- The worker uses one Gemini Live session per room job; `AgentServer` isolates
  jobs per process, so concurrent rooms scale across cores.
