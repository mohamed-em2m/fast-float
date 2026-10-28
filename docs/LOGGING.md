# Logging

One engine configures every process in the stack: the FastAPI server, the GPU
inference worker, the Gemini/LiveKit agent, the Gradio apps and the CLI
supervisor. The implementation lives in `fast_float/logging_utils/`;
`fast_float/utils/logger.py` is a thin backwards-compatible facade, so existing
`from fast_float.utils.logger import logger` imports keep working.

```python
from fast_float.logging_utils import configure_logging

configure_logging(role="api")  # once, in the entry point
```

Everything else just uses `logging.getLogger(__name__)`.

## What you get

| Sink | Default | Format |
| --- | --- | --- |
| `stderr` | on, colors only on a TTY | human-readable |
| `logs/<role>.log` | on, 50 MiB x 5 | JSONL (one object per line) |
| `logs/combined.log` | supervisor / shell scripts | prefixed console text |
| `logs/server.log` | hypercorn via `config/logging.json` | JSONL, 20 MiB x 3 |
| `logs/crash-<role>-<pid>.log` | on for long-running roles | faulthandler dump |

Console line:

```text
2026-10-28 02:00:35.654 | INFO     | api:36256 | fast_float.http | [req=abc123 sess=s1 gen=3] GET /health 200 1.2ms
```

JSONL line:

```json
{"ts":"2026-10-27T23:00:35.654Z","level":"INFO","logger":"fast_float.http","role":"api","pid":36256,"thread":"MainThread","msg":"GET /health 200 1.2ms","req_id":"abc123","session_id":"s1","gen":3}
```

Stable keys: `ts` (UTC, ISO-8601), `level`, `logger`, `role`, `pid`, `thread`,
`msg`. Optional: `req_id`, `session_id`, `gen`, `room`, any `extra={...}`
fields, `exc` (formatted traceback) and `stack`.

## Roles and files

`configure_logging(role=...)` selects the file name and the `role` column.
Entry points pass it explicitly:

| Entry point | Role | File |
| --- | --- | --- |
| `fast_float.main` (hypercorn/uvicorn) | `api` | `logs/api.log` |
| `fast_float.workers.inference_worker` | `worker` | `logs/worker.log` |
| `fast_float.livekit.gemini.agent` | `agent` | `logs/agent.log` |
| `fast_float.interfaces.float_live_app` | `gradio` | `logs/gradio.log` |
| `fast_float.cli` supervisor | `supervisor` | `logs/supervisor.log` |
| `pytest` | `test` | disabled unless requested |

`resolve_role()` falls back to `FLOAT_PROCESS_ROLE`, then to `sys.argv`
(`pytest`, `inference_worker`, `gemini.agent`, `float_live_app`, `hypercorn`,
`fast_float.cli`), then to `app`.

## Configuration

Precedence: explicit arguments > environment > `[logging]` in
`fast_float/config/settings.toml` > built-in defaults.

| Environment | TOML | Default | Meaning |
| --- | --- | --- | --- |
| `FLOAT_LOG_LEVEL` | `level` | `INFO` | level name or number |
| `FLOAT_DEBUG`, `FLOAT_TIMING` | — | off | truthy -> `DEBUG` |
| `FLOAT_PROCESS_ROLE` | — | inferred | role / default file name |
| `FLOAT_LOG_DIR` | `dir` | `<repo>/logs` | directory for all files |
| `FLOAT_LOG_FILE` | — | `<role>.log` | file path/name; relative paths are under `FLOAT_LOG_DIR`; `none`/`0`/`off` disables |
| `FLOAT_LOG_TO_FILE` | `file` | on (off for tests) | `false` = console only |
| `FLOAT_LOG_JSON` | `json` | `true` | `false` = text in the file |
| `FLOAT_LOG_CONSOLE` | `console` | `true` | `false` = file only |
| `FLOAT_LOG_COLOR` | `color` | auto | `false` = no ANSI (also `NO_COLOR=1`) |
| `FLOAT_LOG_MAX_BYTES` | `max_bytes` | `52428800` | rotation size |
| `FLOAT_LOG_BACKUPS` | `backup_count` | `5` | rotated files kept |
| `FLOAT_LOG_RATE_LIMIT` | `rate_limit` | `true` | collapse repeats on console |
| `FLOAT_LOG_QUIET` | `quiet` | `true` | quiet noisy third-party loggers |
| `FLOAT_LOG_UNIFY_SERVERS` | `unify_servers` | `true` | hypercorn/uvicorn share the sinks |
| `FLOAT_LOG_SHARED` | `shared` | `false` | one locked file for all processes |
| `FLOAT_LOG_TEE` | — | `1` | supervisor/scripts write `combined.log` (`0`/`false`/`no`/`off` disables) |

`configure_logging()` is idempotent: repeating it with the same structural
options only updates the level. Reconfiguring a different role/file/sink closes
the previous handlers and installs new ones; handlers that were not installed by
the engine are left alone.

## Request correlation

`RequestContextMiddleware` (pure ASGI, no `BaseHTTPMiddleware`) binds
`X-Request-ID` and `X-Session-ID` from the request headers, generates a short id
when `X-Request-ID` is missing, and echoes it back on the response. Every record
emitted while handling the request carries `req_id`/`session_id`.

The worker does not receive HTTP headers. `inference_worker.py` binds the ids
from the incoming `IPCRequest`:

```python
with log_context(req_id=req.req_id, session_id=req.session_id, gen=req.gen):
    handle(req)
```

`contextvars` are copied into `asyncio.create_task` and `asyncio.to_thread`, so
anything spawned inside the request keeps the ids. `run_in_executor` does not
copy them; bind explicitly at the entry point.

## Redaction

Both formatters mask credentials before writing:

* `key=value` / `key: value` / JSON keys such as `api_key`, `token`, `secret`,
  `password`, `authorization`, `livekit_api_secret`, `gemini_api_key`
  (structured `extra` fields named like these become `***` entirely),
* `Authorization: Bearer <token>`,
* query parameters `?access_token=...`, `?token=...`, `?api_key=...`,
* Google API keys (`AIza...`).

Message strings are truncated at 8 KiB (extras at 4 KiB, lists/dicts at 50
items) so a bad `repr()` cannot fill the disk. Redaction is best-effort; never
log a credential on purpose.

## Rotation and multiple processes

Each process owns its own file (`logs/api.log`, `logs/worker.log`, ...), so
`RotatingFileHandler` is safe: there is exactly one writer per file.

`logs/combined.log` is written by a **single** writer:

* the CLI supervisor (`fast_float.cli`) tees each child's stdout/stderr from
  reader threads, guarded by one `threading.Lock`;
* the shell scripts do `exec > >(tee -a logs/combined.log) 2>&1`.

No cross-process lock is involved. Disable with `FLOAT_LOG_TEE=0`.

If you really need several processes appending to one file, set
`FLOAT_LOG_SHARED=1` (or `[logging] shared = true`). That selects
`LockedFileHandler` (`logs/shared.log`) which serializes each record with a
thread lock plus an OS file lock on `shared.log.lock`:

* POSIX `fcntl.flock`, Windows `msvcrt.locking`;
* lock timeout 250 ms (`timeout=`); on timeout the record is still written
  unlocked and counted in `handler.unlocked_writes`;
* **no rotation** — concurrent `RotatingFileHandler` instances would race during
  rollover. Use dated file names or an external rotator instead.

For most deployments the per-process files plus the supervisor tee are the
recommended setup.

Multi-worker caveat: a server started with more than one worker process (for
example `uvicorn --workers 4`) means several processes share `logs/api.log`.
`RotatingFileHandler` is only safe with one writer, so either run a single
worker (the default here) or enable `FLOAT_LOG_SHARED` / give each worker its
own file.

## Crash diagnostics

Long-running roles install `faulthandler` (all threads) writing to
`logs/crash-<role>-<pid>.log`, plus hooks for uncaught exceptions in the main
thread, worker threads, asyncio tasks and `__del__`. A CUDA segfault therefore
leaves a Python stack in the crash file instead of nothing.

## Server logs

Hypercorn configures its `Logger` lazily, after importing the app. Entry points
call `reroute_server_loggers()` in the lifespan startup so server records use the
app's console/file handlers. The CLI and the shell scripts additionally pass:

```bash
hypercorn fast_float.main:app --log-config json:config/logging.json
```

`config/logging.json` writes `logs/server.log` (relative to the working
directory, so run from the repo root) and keeps health-check access lines out of
both sinks. Access logging is enabled explicitly (`accesslog = "-"` /
`--access-logfile -`); without it hypercorn never calls its access logger. Do
not point this config at `logs/api.log`: two `RotatingFileHandler` instances on
one path race during rollover.

## Recipes

```bash
# Follow the human-readable timeline
tail -f logs/combined.log

# Follow one process
tail -f logs/api.log | jq -r '"\(.ts) \(.level) \(.logger) \(.msg)"'

# All warnings/errors across processes
jq -c 'select(.level == "ERROR" or .level == "CRITICAL")' logs/*.log

# One live session (API + worker + agent)
jq -c 'select(.req_id == "abc123" or .session_id == "s1")' logs/*.log

# Slowest requests (if you add duration extras)
jq -s 'sort_by(.duration_ms) | reverse | .[:20]' logs/api.log
```

PowerShell: `Get-Content logs\api.log -Wait`; `Get-Content logs\api.log | ConvertFrom-Json`.

## Tests

```bash
uv run pytest tests/test_logging.py -q
```

Covered: level precedence, idempotent reconfiguration, handler ownership,
JSON schema/exceptions, ANSI-free output, context propagation (sync + asyncio),
redaction, rate limiting, rotation and multi-process line integrity for
`LockedFileHandler`.

`configure_logging(role="test")` disables the file sink by default regardless of
`settings.toml`; pass `file=True` to opt in. Tests never write into the
repository's `logs/` directory unless they ask for it.

## Troubleshooting

* **Nothing in a file** — check `current_log_file()` / the startup banner
  (`Logs: role=... file=...`) and `FLOAT_LOG_TO_FILE`.
* **Duplicate lines** — a logger has both its own handler and `propagate=True`.
  The engine sets `propagate=False` on managed roots; a library that also adds a
  root handler will double-print. Set `unify`/handlers accordingly.
* **No colors** — expected when stderr is redirected (CI, `tee`, systemd) or
  `NO_COLOR` is set.
* **Permission denied on `logs/`** - crash dumps are skipped with a warning; the
  console and any writable per-process file keep working.

## Examples

Configure once in the entry point, then use `logging.getLogger(__name__)`:

```python
import logging

from fast_float.logging_utils import configure_logging, log_context

configure_logging(role="api")  # console + logs/api.log (JSONL)
log = logging.getLogger(__name__)

log.info("plain line")

with log_context(req_id="abc123", session_id="s1", gen=3):
    log.info("correlated line")  # console: [req=abc123 sess=s1 gen=3]
```

Run the stack and read the files it produces:

```bash
FLOAT_DEBUG=1 ./scripts/run_dual_process.sh
ls logs/        # api.log worker.log supervisor.log combined.log server.log

jq -c 'select(.req_id == "abc123")' logs/*.log
jq -r 'select(.level == "ERROR") | "\(.ts) \(.role) \(.msg)"' logs/*.log
tail -f logs/combined.log
```

Change behaviour without touching code:

```bash
FLOAT_LOG_LEVEL=DEBUG ./scripts/run_api.sh       # debug level only
FLOAT_LOG_CONSOLE=0 ./scripts/run_api.sh         # file only
FLOAT_LOG_DIR=/var/log/float ./scripts/run_api.sh
FLOAT_LOG_JSON=false ./scripts/run_api.sh        # text instead of JSONL
FLOAT_LOG_SHARED=1 ./scripts/run_api.sh          # several writers, one locked file
```

PowerShell:

```powershell
Select-String -Path logs\*.log -Pattern '"level":"ERROR"'
Get-Content logs\combined.log -Wait
```

`from fast_float.utils.logger import logger` and `set_log_level("DEBUG")` keep
working through the facade. Run modes and ports: [USAGE.md](USAGE.md).
