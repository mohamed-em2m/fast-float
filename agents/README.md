# FLOAT Gemini Live voice worker

A [LiveKit Agents](https://docs.livekit.io/agents/) worker that runs the
conversation with **Gemini Live** (speech-to-speech) and drives the FLOAT
avatar. It lives in its own uv project because `livekit-agents` requires
newer `numpy`, `pydantic` and `websockets` than FLOAT's pinned inference
stack; the two processes talk over LiveKit, not Python imports.

```
Browser mic ──WebRTC──▶ LiveKit SFU ──▶ Gemini Live worker (this project)
                              │                 │  TTS PCM over DataStream
                              │                 ▼
                              │        FLOAT avatar (main repo)
                              │        FLOAT renderer → avatar_video + avatar_audio
                              └──────────────▶ Browser playback
```

## Run

```bash
cd agents
uv sync
# Values come from ../fast_float/config/secerts.toml + settings.toml; only
# copy the env file on a box without the shared config.
cp .env.example .env      # optional fallback
uv run python -m float_agent.agent start
```

Use `start`, not the deprecated `dev` command: livekit-agents removed
in-process auto-reload, so `dev` now only prints deprecation notices.

A healthy worker logs two lines at startup:

```
worker starting: agent_name=float-gemini url=wss://… config_dir=…/fast_float/config livekit_credentials=True gemini_key=True model=…
registered worker  (livekit.agents)  agent_name=float-gemini url=wss://…
```

`worker starting` shows which SFU, agent name and config directory the
process actually resolved; `registered worker` confirms the SFU accepted it.
No `registered worker` line ⇒ the API's dispatch reports `NO job`.

The worker registers as `float-gemini`. It stays idle until the FLOAT API
dispatches it into a room (which happens automatically when the Studio starts
a **Gemini Live** conversation, as long as `FLOAT_GEMINI_AUTODISPATCH` is on).

For a production process use `start` instead of `dev`:

```bash
uv run python -m float_agent.agent start
```

## Flow

1. Browser starts a conversation against the FLOAT API with
   `agent: "gemini"`.
2. FLOAT connects `float-avatar` (kind `AGENT`, so LiveKit Agents' RoomIO
   ignores its synthesized voice) and dispatches this worker.
3. The worker joins, listens to the browser's microphone track, and Gemini
   replies with audio.
4. `DataStreamAudioOutput` streams each TTS segment to `float-avatar` on the
   `lk.audio_stream` topic and waits for `lk.playback_started`.
5. FLOAT renders the avatar at 25 fps, publishes synchronized
   `avatar_video` + `avatar_audio`, and reports `lk.playback_finished`.
6. Barge-in travels the other way: `lk.clear_buffer` stops FLOAT's queued
   audio/video immediately.

## Voice tools

| tool | action |
| --- | --- |
| `float_health` | `GET /health` on the FLOAT API |
| `float_gpu` | `GET /gpu` |
| `generation_progress` | `GET /progress/{request_id}` |
| `conversation_status` | `GET /livekit/conversation/status?room=…` |
| `end_conversation` | `POST /livekit/conversation/stop`, then close the session |

The system prompt is in `float_agent/config.py`; override it with
`FLOAT_AGENT_INSTRUCTIONS`.

## Configuration

`fast_float/config/secerts.toml` (secrets) and `settings.toml` (defaults) are
the **single source of truth**: both the FLOAT API and this worker read the
same file sections, and a file value wins over `agents/.env` / plain env. An
env copy that *disagrees* with the file is ignored and logged as stale — a
mismatch there is what makes the worker register on a different LiveKit
project than the API dispatches into. Env values are only the fallback for
boxes without the shared config:

| shared place | keys |
| --- | --- |
| `fast_float/config/secerts.toml` `[gemini]` | `api_key` (and `[livekit]` url/key/secret) |
| `fast_float/config/settings.toml` `[gemini]` | `model`, `voice`, `language`, `vertexai`, `project`, `location`, `avatar_identity`, `agent_name`, `autodispatch` |

Override the config directory with `FLOAT_CONFIG_DIR` (defaults to the
sibling `fast_float/config` of this repo).

A key and/or model name typed into the Studio's header **Config** panel
travels in the dispatch `metadata` (`{"gemini_api_key": "...",
"gemini_model": "..."}`) and wins for that session (see
`dispatch_config()` in `float_agent/config.py`); the Studio key never
appears in any log.

| env | default | meaning |
| --- | --- | --- |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | shared `[livekit]` | SFU credentials (required) |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | shared `[gemini].api_key` | Gemini Developer API key (required unless Vertex AI) |
| `GEMINI_MODEL` | plugin default | Live API model; useful for pinning a preview |
| `GEMINI_VOICE` | `Puck` | Gemini prebuilt voice |
| `GEMINI_LANGUAGE` | auto | BCP-47 language hint |
| `GOOGLE_GENAI_USE_VERTEXAI` / `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` | off | Vertex AI instead of the Developer API |
| `FLOAT_AVATAR_IDENTITY` | `float-avatar` | destination identity for the TTS stream |
| `FLOAT_GEMINI_AGENT_NAME` | `float-gemini` | LiveKit agent name |
| `FLOAT_API_URL` | `http://127.0.0.1:8000` | FLOAT API for tools |
| `FLOAT_API_TIMEOUT_S` | `8` | tool HTTP timeout |
| `FLOAT_AGENT_INSTRUCTIONS` | built-in | override the system prompt |

## Notes

- Keep this process on a host that can reach both the LiveKit SFU and the
  FLOAT API. It does **not** need a GPU.
- `uv sync` creates `agents/.venv`; it is independent from the repository's
  root `.venv`.
- The worker uses one Gemini Live session per room job. `AgentServer` runs
  jobs in separate processes by default, so multiple rooms scale across
  cores.
