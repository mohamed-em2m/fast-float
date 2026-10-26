# Gemini Live voice agent (LiveKit Agents)

FLOAT can run its live avatar conversation with **Gemini Live** — Google's
real-time speech-to-speech model — instead of the in-process agent. A
[LiveKit Agents](https://docs.livekit.io/agents/) worker owns the
conversation, Gemini answers in its own voice, and FLOAT keeps rendering the
lip-synced avatar. The worker is part of this project
([`fast_float/livekit/gemini/`](../fast_float/livekit/gemini/)) and shares its
venv, lockfile and config.

## Why a separate process

LiveKit's dispatch model requires a long-lived worker that registers with the
SFU and receives jobs; the API cannot host it. The worker therefore runs as its
own process, but from the same project — one `uv sync`, one lockfile:

```
Browser mic ──WebRTC──▶ LiveKit SFU ──▶ Gemini Live worker (fast_float/livekit/gemini)
                        │                     │  TTS PCM over DataStream
                        │                     ▼
                        │            FLOAT avatar worker (same project)
                        │            FLOAT renderer → avatar_video + avatar_audio
                        └─────────────────▶ Browser playback
```

`livekit-agents[google]` used to conflict with FLOAT's inference stack; the
dependency matrix now resolves both (see
[`docs/LIVEKIT_INTEGRATION.md`](LIVEKIT_INTEGRATION.md#dependency-matrix)),
so the worker needs no separate environment.

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
[`function_tool`s](../fast_float/livekit/gemini/agent.py): `float_health`,
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
| `fast_float/config/settings.toml` `[gemini]` | `model`, `voice`, `language`, `instructions_file`, `vertexai`… |
| `fast_float/config/settings.toml` `[float]` | all FLOAT model/inference options; `livekit_nfe` is the avatar's sampling budget |
| `prompts/gemini_agent.md` | the spoken persona (`[gemini].instructions_file`) |

The worker's spoken persona is the Markdown file `prompts/gemini_agent.md`,
referenced by `[gemini].instructions_file`; edit the file rather than the
Python constant. An inline `[gemini].instructions` still works as a fallback,
and `FLOAT_AGENT_INSTRUCTIONS` is the last-resort env fallback for boxes
without the shared file. The avatar renderer's speed knob is
`[float].livekit_nfe` (5 fastest … 25 best quality, built-in fallback 6); lower
it if the log reports `[livekit-gen] slower than real time`.

The Studio can also supply **per-session overrides** from its header
**Config** panel (stored in the browser only): a Gemini API key and/or a
Live model name. They travel with the `/livekit/conversation/start` request
and are forwarded to the worker inside the dispatch `metadata`
(`{"gemini_api_key": "...", "gemini_model": "..."}`); the worker prefers
them over its own config for that session and never logs the key. This is a
convenience for quick sessions — for persistent values keep using
`secerts.toml` / `settings.toml` or the `GOOGLE_API_KEY` / `GEMINI_MODEL`
env vars.

Worker-side (see [`.env.example`](../.env.example)):
`LIVEKIT_URL` + `LIVEKIT_API_KEY` + `LIVEKIT_API_SECRET`, `GOOGLE_API_KEY`
(or Vertex AI flags), and optional `GEMINI_MODEL` / `GEMINI_VOICE` /
`GEMINI_LANGUAGE`. The shared FLOAT config is the **single source of truth**:
both processes read the same `secerts.toml` / `settings.toml` sections and a
file value wins over the env; a disagreeing env copy is ignored and logged
as stale. Only set these in the repo-root `.env` when the box has no shared
config — a copy here that drifts is exactly why a worker can register on a
different LiveKit project than the API dispatches into.

## Run it

```bash
# 1. LiveKit SFU (see docker-compose.livekit.yml)
# 2. FLOAT API (this repo)
# 3. the Gemini worker
./scripts/run_gemini_agent.sh
# or: uv run python -m fast_float.livekit.gemini.agent start
```

`run_all.sh` starts the worker too when LiveKit and Gemini are both configured
(`FLOAT_GEMINI_WORKER=auto|yes|no`, default `auto`). Set `FLOAT_CONFIG_DIR` on
both processes if the checkout is not the default location.

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
LiveKit SFU, the Gemini worker (`fast_float/livekit/gemini/`) and the FLOAT
avatar worker (same project, separate task). The sequence below is what happens
between *Go Live* and the first frame on screen:

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
| `GOOGLE_API_KEY`, `LIVEKIT_*` env / root `.env` | defaults | fallback for boxes without the shared config |

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
- **Model names are validated at the boundary — including the configured
  one.** `RealtimeModel` accepts any string and only fails on the Gemini
  websocket seconds later, so the worker normalizes case/quotes, strips a
  provider prefix (`google/gemini-3.8-live` → `gemini-3.8-live`, which
  otherwise becomes the nonexistent path `models/google/…` and dies with
  `1008 policy violation`), and warns about unknown-but-plausible ids
  instead of starting a session that cannot connect. `settings.toml`'s
  `model` goes through the same function as the Studio override: use the
  bare id, e.g. `gemini-3.8-live`.

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
  `lk.clear_buffer`; the agent keeps its own turn state. Because avatar audio
  is streamed to the renderer as it is produced, the output cannot pause or
  rewind, so the worker disables `resume_false_interruption` (upstream would
  otherwise warn that it is ignored on every session).
- The worker uses one Gemini Live session per room job; `AgentServer` isolates
  jobs per process, so concurrent rooms scale across cores.
