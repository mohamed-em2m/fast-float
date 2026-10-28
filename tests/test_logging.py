# SPDX-License-Identifier: Apache-2.0
"""Tests for the unified logging engine (``fast_float.logging_utils``)."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import logging.handlers
import subprocess
import sys
from pathlib import Path

import pytest

from fast_float.logging_utils import (
    ConsoleFormatter,
    ContextFilter,
    JsonFormatter,
    LockedFileHandler,
    RateLimitFilter,
    access_log_enabled,
    configure_logging,
    current_context,
    current_log_file,
    log_context,
    redact_text,
    reroute_server_loggers,
    resolve_level,
)
from fast_float.logging_utils import config as logconfig
from fast_float.logging_utils.uvicorn_logging import create_hypercorn_log_config

_ROOT = Path(__file__).resolve().parents[1]
_MANAGED = ("fast_float", "float_gemini", "float_live_app")

_LOG_ENV = (
    "FLOAT_DEBUG",
    "FLOAT_TIMING",
    "FLOAT_LOG_LEVEL",
    "FLOAT_LOG_DIR",
    "FLOAT_LOG_FILE",
    "FLOAT_LOG_TO_FILE",
    "FLOAT_LOG_SHARED",
    "FLOAT_LOG_JSON",
    "FLOAT_LOG_CONSOLE",
    "FLOAT_LOG_COLOR",
    "FLOAT_LOG_MAX_BYTES",
    "FLOAT_LOG_BACKUPS",
    "FLOAT_LOG_RATE_LIMIT",
    "FLOAT_LOG_QUIET",
    "FLOAT_LOG_UNIFY_SERVERS",
    "FLOAT_PROCESS_ROLE",
)


@pytest.fixture(autouse=True)
def _clean_logging(monkeypatch):
    """Give every test a blank logging state and no FLOAT_LOG_* overrides."""
    for name in _LOG_ENV:
        monkeypatch.delenv(name, raising=False)
    logging.getLogger().handlers.clear()
    logconfig._reset_for_tests()
    yield
    logconfig._reset_for_tests()
    for name in logconfig.SERVER_LOGGERS:
        logging.getLogger(name).handlers = []
        logging.getLogger(name).filters.clear()
    logging.getLogger().handlers.clear()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _capture_logger(name: str = "fast_float.capture") -> tuple[logging.Logger, list[logging.LogRecord]]:
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger(name)
    logger.handlers = [_ListHandler()]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, records


# --------------------------------------------------------------------------- #
# Level / env resolution
# --------------------------------------------------------------------------- #
def test_level_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOAT_LOG_LEVEL", "WARNING")
    assert resolve_level() == logging.WARNING

    # FLOAT_LOG_LEVEL wins over FLOAT_DEBUG.
    monkeypatch.setenv("FLOAT_DEBUG", "1")
    assert resolve_level() == logging.WARNING

    # An explicit argument wins over both.
    assert configure_logging(level="ERROR", role="test", file=False, crash=False) == logging.ERROR

    monkeypatch.delenv("FLOAT_LOG_LEVEL")
    assert resolve_level() == logging.DEBUG

    # And debug=True outranks everything.
    assert configure_logging(debug=True, level="ERROR", role="test", file=False, crash=False) == logging.DEBUG


def test_access_logging_is_opt_in(monkeypatch):
    """Access lines are one record per request, so they must not be the default."""
    monkeypatch.delenv("FLOAT_ACCESS_LOG", raising=False)
    assert access_log_enabled() is False

    for off in ("", "0", "false", "no", "off"):
        monkeypatch.setenv("FLOAT_ACCESS_LOG", off)
        assert access_log_enabled() is False, off

    for on in ("1", "true", "yes", "on"):
        monkeypatch.setenv("FLOAT_ACCESS_LOG", on)
        assert access_log_enabled() is True, on


def test_configure_is_idempotent(tmp_path):
    kwargs = dict(role="test", log_dir=tmp_path, file=True, console=True, crash=False)
    configure_logging(**kwargs)
    first = logging.getLogger("fast_float").handlers
    configure_logging(**kwargs)
    second = logging.getLogger("fast_float").handlers
    assert first == second
    assert len(first) == 2  # console + file


def test_set_log_level_updates_handlers(tmp_path):
    configure_logging(level="INFO", role="test", log_dir=tmp_path, file=True, console=True, crash=False)
    from fast_float.logging_utils import set_log_level

    set_log_level("DEBUG")
    assert logging.getLogger("fast_float").level == logging.DEBUG
    assert all(handler.level == logging.DEBUG for handler in logging.getLogger("fast_float").handlers)


def test_foreign_handlers_are_preserved(tmp_path):
    marker = logging.NullHandler()
    managed = logging.getLogger("fast_float")
    managed.addHandler(marker)
    try:
        configure_logging(role="test", log_dir=tmp_path, file=True, console=True, crash=False)
        assert marker in managed.handlers
        assert managed.propagate is False
    finally:
        managed.removeHandler(marker)


def test_reconfigure_swaps_owned_handlers(tmp_path):
    configure_logging(role="test", log_dir=tmp_path, file=True, console=True, crash=False)
    old = tuple(logging.getLogger("fast_float").handlers)
    configure_logging(role="test", log_dir=tmp_path / "other", file=True, console=True, crash=False)
    new = tuple(logging.getLogger("fast_float").handlers)
    assert set(old).isdisjoint(new)
    assert all(handler._float_owned for handler in new)


def test_reroute_server_loggers_adopts_unconfigured_server(tmp_path):
    configure_logging(role="test", log_dir=tmp_path, file=True, console=True, crash=False)
    server = logging.getLogger("hypercorn.error")
    server.handlers = []
    reroute_server_loggers()
    assert set(logging.getLogger("fast_float").handlers) == set(server.handlers)
    assert server.propagate is False


def test_reroute_leaves_explicit_server_config_alone(tmp_path):
    configure_logging(role="test", log_dir=tmp_path, file=True, console=True, crash=False)
    server = logging.getLogger("hypercorn.error")
    explicit = logging.StreamHandler(io.StringIO())
    explicit.setFormatter(JsonFormatter())
    server.handlers = [explicit]
    reroute_server_loggers()
    assert server.handlers == [explicit]


def test_reconfigure_keeps_explicit_server_config(tmp_path):
    configure_logging(role="test", log_dir=tmp_path, file=True, console=True, crash=False)
    server = logging.getLogger("hypercorn.error")
    explicit = logging.StreamHandler(io.StringIO())
    explicit.setFormatter(JsonFormatter())
    server.handlers = [explicit]

    configure_logging(role="test", log_dir=tmp_path / "new", file=True, console=True, crash=False)
    assert server.handlers == [explicit]


def test_generated_hypercorn_config_has_context_and_levels(tmp_path):
    config = create_hypercorn_log_config(
        excluded_paths=["/health"],
        log_file=str(tmp_path / "server.log"),
        role="api",
    )
    assert config["filters"]["context"]["role"] == "api"
    assert config["loggers"]["hypercorn.access"]["level"] == "INFO"
    assert "file" in config["handlers"]
    assert "context" in config["handlers"]["file"]["filters"]


# --------------------------------------------------------------------------- #
# Formats
# --------------------------------------------------------------------------- #
def test_json_schema_and_exception(tmp_path):
    configure_logging(
        role="test",
        log_dir=tmp_path,
        file=True,
        console=False,
        json_file=True,
        crash=False,
    )
    path = current_log_file()
    assert path is not None

    logger = logging.getLogger("fast_float.json_test")
    logger.info("hello %s", "world")
    with log_context(req_id="req-1", session_id="sess-1", gen=2):
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("failed")
    for handler in logging.getLogger("fast_float").handlers:
        handler.flush()

    rows = _read_jsonl(path)
    assert len(rows) == 2

    first = rows[0]
    assert first["level"] == "INFO"
    assert first["logger"] == "fast_float.json_test"
    assert first["role"] == "test"
    assert first["msg"] == "hello world"
    assert first["ts"].endswith("Z")
    assert isinstance(first["pid"], int)
    assert "req_id" not in first

    second = rows[1]
    assert second["req_id"] == "req-1"
    assert second["session_id"] == "sess-1"
    assert second["gen"] == 2
    assert "ValueError: boom" in second["exc"]


def test_console_has_no_ansi_when_color_is_off():
    record = logging.LogRecord("fast_float.x", logging.INFO, __file__, 1, "hello", (), None)
    line = ConsoleFormatter(color=False).format(record)
    assert "\x1b[" not in line
    assert "INFO" in line
    assert line.endswith("hello")

    colored = ConsoleFormatter(color=True).format(record)
    assert "\x1b[" in colored


def test_json_formatter_handles_standalone_record():
    record = logging.LogRecord("fast_float.x", logging.WARNING, __file__, 1, "plain", (), None)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "plain"
    assert payload["role"] == "app"


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
def test_context_propagates_to_records_and_resets():
    logger, records = _capture_logger()
    logger.addFilter(ContextFilter("test"))
    try:
        logger.info("outside")
        with log_context(req_id="r-9", session_id="s-9", room="room-1"):
            logger.info("inside")
        logger.info("after")
    finally:
        logger.filters.clear()

    assert records[0].req_id == ""
    assert records[1].req_id == "r-9"
    assert records[1].session_id == "s-9"
    assert records[1].room == "room-1"
    assert records[1].context == "[req=r-9 sess=s-9 room=room-1] "
    assert records[2].req_id == ""


def test_context_is_copied_into_async_tasks():
    async def scenario() -> tuple[str, str]:
        inside = ""
        outside = ""
        with log_context(req_id="async-1"):

            async def child() -> str:
                await asyncio.sleep(0)
                return current_context().req_id

            inside = await asyncio.create_task(child())
        outside = current_context().req_id
        return inside, outside

    assert asyncio.run(scenario()) == ("async-1", "")


def test_explicit_record_fields_win_over_context():
    logger, records = _capture_logger()
    logger.addFilter(ContextFilter("test"))
    try:
        with log_context(req_id="from-context"):
            logger.info("x", extra={"req_id": "from-call"})
    finally:
        logger.filters.clear()
    assert records[0].req_id == "from-call"


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "secret"),
    [
        ("api_key=SECRET123", "SECRET123"),
        ("token: abcdef", "abcdef"),
        ("Authorization: Bearer abc.def-ghi", "abc.def-ghi"),
        ("https://x.test/cb?access_token=deadbeef&x=1", "deadbeef"),
        ("AIza" + "a" * 35, "AIza" + "a" * 35),
    ],
)
def test_redact_text(raw, secret):
    redacted = redact_text(raw)
    assert secret not in redacted
    assert "***" in redacted


def test_json_formatter_redacts_structured_fields(tmp_path):
    configure_logging(
        role="test",
        log_dir=tmp_path,
        file=True,
        console=False,
        crash=False,
    )
    path = current_log_file()
    logging.getLogger("fast_float.redact").info(
        "auth",
        extra={"api_key": "SECRET", "nested": {"password": "hunter2"}},
    )
    for handler in logging.getLogger("fast_float").handlers:
        handler.flush()
    row = _read_jsonl(path)[0]
    assert row["api_key"] == "***"
    assert row["nested"]["password"] == "***"


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def test_rate_limit_collapses_repeats():
    filt = RateLimitFilter(burst=2, window_s=60)

    def record() -> logging.LogRecord:
        return logging.LogRecord("fast_float.noisy", logging.WARNING, __file__, 1, "same %s", ("line",), None)

    records = [record() for _ in range(5)]
    passed = [filt.filter(row) for row in records]
    assert passed == [True, True, True, False, False]
    assert "suppressed" in records[2]._console_note
    assert records[2].msg == "same %s"  # the file handler still sees the original


def test_console_formatter_renders_rate_limit_note():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "same", (), None)
    record._console_note = " (further similar records suppressed for 10s)"
    line = ConsoleFormatter(color=False).format(record)
    assert "same (further similar records suppressed for 10s)" in line


def test_rate_limit_allows_different_messages():
    filt = RateLimitFilter(burst=1, window_s=60)
    assert filt.filter(logging.LogRecord("x", logging.INFO, __file__, 1, "a", (), None))
    assert filt.filter(logging.LogRecord("x", logging.INFO, __file__, 1, "b", (), None))


# --------------------------------------------------------------------------- #
# Rotation & multi-process
# --------------------------------------------------------------------------- #
def test_file_handler_rotates(tmp_path):
    configure_logging(
        role="test",
        log_dir=tmp_path,
        file=True,
        console=False,
        max_bytes=150,
        backup_count=2,
        crash=False,
    )
    logger = logging.getLogger("fast_float.rotate")
    for index in range(20):
        logger.info("line %02d %s", index, "x" * 60)
    for handler in logging.getLogger("fast_float").handlers:
        handler.flush()

    files = sorted(tmp_path.glob("test.log*"))
    assert len(files) >= 2
    assert any(name.name.endswith(".1") for name in files)


def test_locked_file_handler_keeps_lines_intact(tmp_path):
    path = tmp_path / "shared.log"
    lines_per_process = 40
    processes = 3

    code = """
import json, logging, sys
from fast_float.logging_utils.handlers import LockedFileHandler

path = sys.argv[1]
tag = sys.argv[2]
handler = LockedFileHandler(path, timeout=10.0)
handler.setFormatter(logging.Formatter("%(message)s"))
logger = logging.getLogger("mp-" + tag)
logger.handlers = [handler]
logger.propagate = False
logger.setLevel(logging.INFO)
for i in range(__COUNT__):
    logger.info(json.dumps({"tag": tag, "i": i}))
handler.close()
""".replace("__COUNT__", str(lines_per_process))

    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(path), f"p{i}"],
            cwd=_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for i in range(processes)
    ]
    for proc in procs:
        _, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err.decode("utf-8", "replace")

    rows = _read_jsonl(path)
    assert len(rows) == processes * lines_per_process
    seen = {(row["tag"], row["i"]) for row in rows}
    assert len(seen) == processes * lines_per_process


def test_locked_file_handler_counts_timeout_writes(tmp_path):
    handler = LockedFileHandler(tmp_path / "shared.log")
    try:
        assert handler.timeout == 0.25
        assert handler.lock_path.endswith(".lock")
        handler.emit(logging.LogRecord("x", logging.INFO, __file__, 1, "one", (), None))
        assert (tmp_path / "shared.log").read_text(encoding="utf-8") == "one\n"
        assert handler.unlocked_writes == 0
    finally:
        handler.close()


# --------------------------------------------------------------------------- #
# Settings / facade
# --------------------------------------------------------------------------- #
def test_settings_log_dir_is_used(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOAT_LOG_DIR", str(tmp_path / "from-env"))
    monkeypatch.setenv("FLOAT_LOG_TO_FILE", "1")
    configure_logging(role="test", console=False, crash=False)
    assert current_log_file() == tmp_path / "from-env" / "test.log"
    assert (tmp_path / "from-env").is_dir()


def test_explicit_file_disables_file_logging(tmp_path):
    configure_logging(role="test", log_dir=tmp_path, file=False, console=True, crash=False)
    assert current_log_file() is None
    assert not list(tmp_path.glob("*.log"))


def test_setup_logger_adopts_foreign_root(tmp_path):
    from fast_float.utils.logger import setup_logger

    logger = setup_logger("custom_app_root")
    assert logger is logging.getLogger("custom_app_root")
    assert logger.handlers
    assert logger.propagate is False
