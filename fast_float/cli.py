"""CLI entry point for fast-float (vLLM-style runner and supervisor).

Provides commands:
  fast-float serve      Start the backend API server (+ optional worker & frontend)
  fast-float worker     Start only the dedicated GPU inference worker
  fast-float ui         Start only the React Studio frontend dev server
  fast-float agent      Start only the Gemini Live agent
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Serializes writes to the optional combined log from the per-process reader
# threads (one thread per child). This is a single-writer design: the
# supervisor owns the file, so no OS-level lock and no rotation races.
_TEE_LOCK = threading.Lock()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _run_worker(args: argparse.Namespace) -> int:
    """Run dedicated GPU inference worker process."""
    from fast_float.workers.inference_worker import main as worker_main

    if args.control_port:
        os.environ["FLOAT_ZMQ_CONTROL_PORT"] = str(args.control_port)
    if args.frames_port:
        os.environ["FLOAT_ZMQ_FRAMES_PORT"] = str(args.frames_port)
    if args.shm_name:
        os.environ["FLOAT_SHM_NAME"] = str(args.shm_name)
    if args.shm_slots:
        os.environ["FLOAT_SHM_SLOTS"] = str(args.shm_slots)

    worker_main()
    return 0


def _run_ui(args: argparse.Namespace) -> int:
    """Run React Studio frontend dev server."""
    root = _repo_root()
    frontend_dir = root / "frontend"
    if not frontend_dir.exists():
        print(f"❌ Frontend directory not found at {frontend_dir}", file=sys.stderr)
        return 1

    port = str(args.port or 3000)
    host = str(args.host or "0.0.0.0")

    cmd = ["npm", "run", "dev", "--", "--host", host, "--port", port]
    # On Windows npm is npm.cmd
    if sys.platform == "win32":
        cmd[0] = "npm.cmd"

    try:
        proc = subprocess.run(cmd, cwd=frontend_dir, check=False)
        return proc.returncode
    except FileNotFoundError:
        print("❌ npm not found. Please install Node.js and npm.", file=sys.stderr)
        return 1


def _run_agent(args: argparse.Namespace) -> int:
    """Run LiveKit Gemini Live conversational agent."""
    from fast_float.livekit.gemini.agent import main as agent_main

    agent_main()
    return 0


def _stream_process_output(proc: subprocess.Popen, prefix: str, sink=None) -> threading.Thread:
    """Read lines from proc.stdout and print with prefix (optionally tee).

    Returns the reader thread so callers/tests can join it.
    """

    def _reader():
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, b""):
            try:
                decoded = line.decode("utf-8", errors="replace")
            except Exception:
                decoded = str(line)
            # Children on Windows end lines with CRLF; normalize so the tee
            # file (opened with newline="") does not end up with "\r\r\n".
            decoded = decoded.rstrip("\r\n") + "\n"
            text = f"{prefix} {decoded}"
            with _TEE_LOCK:
                sys.stdout.write(text)
                sys.stdout.flush()
                if sink is not None:
                    try:
                        sink.write(text)
                        sink.flush()
                    except Exception:
                        pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return t


def _serve_supervisor(args: argparse.Namespace) -> int:
    """Start and supervise the multi-process stack."""
    from fast_float.logging_utils import configure_logging, current_log_file

    repo = _repo_root()
    procs: list[tuple[str, subprocess.Popen]] = []

    configure_logging(role="supervisor")
    log_file = current_log_file()
    log_dir = log_file.parent if log_file is not None else repo / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PORT"] = str(args.port)
    env["FLOAT_LOG_DIR"] = str(log_dir)
    if args.ngpus:
        env["NGPUS"] = str(args.ngpus)

    # Single-writer combined view of every child's console output. Each
    # process still writes its own JSONL file (logs/<role>.log); this file is
    # the interleaved human-readable timeline and needs no cross-process lock.
    tee_file = None
    tee_enabled = os.environ.get("FLOAT_LOG_TEE", "1").strip().lower() not in {"0", "false", "no", "off"}
    if tee_enabled:
        try:
            tee_file = open(  # noqa: SIM115 - closed in _shutdown()
                log_dir / "combined.log",
                "a",
                encoding="utf-8",
                buffering=1,
                errors="backslashreplace",
                # Lines are normalized to "\n" above; keep the file LF-only so
                # tee'd text does not gain a second CR on Windows.
                newline="",
            )
        except OSError as exc:
            print(f"⚠️  could not open {log_dir / 'combined.log'}: {exc}", file=sys.stderr)

    if args.dual_process:
        env["FLOAT_IPC_ENABLED"] = "true"
        if args.control_port:
            env["FLOAT_ZMQ_CONTROL_PORT"] = str(args.control_port)
        if args.frames_port:
            env["FLOAT_ZMQ_FRAMES_PORT"] = str(args.frames_port)
        if args.shm_name:
            env["FLOAT_SHM_NAME"] = str(args.shm_name)
        if args.shm_slots:
            env["FLOAT_SHM_SLOTS"] = str(args.shm_slots)

    print("=" * 60)
    print("🚀 fast-float inference server")
    print(f"   API Address:      http://{args.host}:{args.port}")
    print(f"   Dual Process:     {'Enabled (ZeroMQ + SHM)' if args.dual_process else 'Disabled (Single Process)'}")
    print(f"   Log directory:    {log_dir}")
    if args.frontend:
        print(f"   Studio Frontend:  http://localhost:{args.frontend_port}")
    print("=" * 60)

    # 1. Start Worker if dual_process is enabled
    if args.dual_process:
        print("🧠 [1/3] Starting FLOAT GPU Inference Worker...")
        worker_env = dict(env, FLOAT_PROCESS_ROLE="worker")
        worker_cmd = [sys.executable, "-m", "fast_float.workers.inference_worker"]
        p_worker = subprocess.Popen(
            worker_cmd,
            cwd=repo,
            env=worker_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        _stream_process_output(p_worker, "[worker]", tee_file)
        procs.append(("worker", p_worker))
        # Wait a moment for worker socket initialization
        time.sleep(2)

    # 2. Start Frontend if requested
    if args.frontend:
        print("🎨 [2/3] Starting Studio Frontend UI...")
        frontend_dir = repo / "frontend"
        npm_bin = "npm.cmd" if sys.platform == "win32" else "npm"
        ui_cmd = [npm_bin, "run", "dev", "--", "--host", "0.0.0.0", "--port", str(args.frontend_port)]
        try:
            p_ui = subprocess.Popen(
                ui_cmd,
                cwd=frontend_dir,
                env=dict(env, FLOAT_PROCESS_ROLE="web"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            _stream_process_output(p_ui, "[web]   ", tee_file)
            procs.append(("frontend", p_ui))
        except FileNotFoundError:
            print("⚠️  npm not found; skipping frontend", file=sys.stderr)

    # 3. Start Backend Server
    print("🌐 [3/3] Starting FastAPI WebRTC API...")
    if args.server_type == "hypercorn":
        server_cmd = [
            sys.executable,
            "-m",
            "hypercorn",
            "fast_float.main:app",
            "--bind",
            f"{args.host}:{args.port}",
            "--worker-class",
            "asyncio",
            "--log-level",
            "info",
            # "-" enables hypercorn's access logger; config/logging.json routes
            # it to logs/server.log and drops health-check noise.
            "--access-logfile",
            "-",
        ]
        log_config = repo / "config" / "logging.json"
        if log_config.exists():
            server_cmd += ["--log-config", f"json:{log_config.as_posix()}"]
    else:
        server_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "fast_float.main:app",
            "--host",
            str(args.host),
            "--port",
            str(args.port),
            "--log-level",
            "info",
        ]

    p_server = subprocess.Popen(
        server_cmd,
        cwd=repo,
        env=dict(env, FLOAT_PROCESS_ROLE="api"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    _stream_process_output(p_server, "[api]   ", tee_file)
    procs.append(("backend", p_server))

    print("\n✅ All components running. Press Ctrl+C to terminate cleanly.\n")

    # Handle graceful shutdown
    def _shutdown(signum, frame):
        print("\n🛑 Shutting down fast-float services...")
        for _name, p in procs:
            try:
                if sys.platform == "win32":
                    p.terminate()
                else:
                    p.send_signal(signal.SIGTERM)
            except Exception:
                pass
        if tee_file is not None:
            try:
                tee_file.flush()
                tee_file.close()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    # Monitor processes
    while True:
        for name, p in procs:
            ret = p.poll()
            if ret is not None:
                print(f"⚠️  Process '{name}' exited with code {ret}")
                _shutdown(None, None)
                return ret
        time.sleep(0.5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fast-float",
        description="fast-float: Real-Time Talking-Face Generation Server & CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- SERVE ---
    p_serve = subparsers.add_parser("serve", help="Launch the fast-float server stack")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    p_serve.add_argument("-p", "--port", type=int, default=8000, help="Port to serve API (default: 8000)")
    p_serve.add_argument(
        "--server-type",
        choices=["hypercorn", "uvicorn"],
        default="hypercorn",
        help="ASGI server implementation (default: hypercorn)",
    )
    p_serve.add_argument(
        "--dual-process",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dual-process architecture (default: enabled). Use --no-dual-process for legacy single-process mode.",
    )
    p_serve.add_argument(
        "--frontend",
        action="store_true",
        help="Concurrently start the React Studio frontend dev server",
    )
    p_serve.add_argument(
        "--frontend-port",
        type=int,
        default=3000,
        help="Port for the Studio frontend (default: 3000)",
    )
    p_serve.add_argument("--ngpus", type=int, default=None, help="Number of GPUs for decoding")
    p_serve.add_argument("--control-port", type=int, default=5555, help="ZeroMQ REP control port (default: 5555)")
    p_serve.add_argument("--frames-port", type=int, default=5556, help="ZeroMQ PUB frames port (default: 5556)")
    p_serve.add_argument("--shm-name", default="float_frame_ring", help="Shared memory segment name")
    p_serve.add_argument("--shm-slots", type=int, default=64, help="Shared memory ring buffer slots (default: 64)")

    # --- WORKER ---
    p_worker = subparsers.add_parser("worker", help="Start only the FLOAT GPU inference worker")
    p_worker.add_argument("--control-port", type=int, default=5555, help="ZeroMQ REP control port (default: 5555)")
    p_worker.add_argument("--frames-port", type=int, default=5556, help="ZeroMQ PUB frames port (default: 5556)")
    p_worker.add_argument("--shm-name", default="float_frame_ring", help="Shared memory ring name")
    p_worker.add_argument("--shm-slots", type=int, default=64, help="Ring buffer slots")

    # --- UI ---
    p_ui = subparsers.add_parser("ui", help="Start only the Studio frontend UI")
    p_ui.add_argument("--host", default="0.0.0.0", help="Host address to bind")
    p_ui.add_argument("-p", "--port", type=int, default=3000, help="Port to serve UI (default: 3000)")

    # --- AGENT ---
    subparsers.add_parser("agent", help="Start the LiveKit Gemini Live conversational agent")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        return _serve_supervisor(args)
    if args.command == "worker":
        return _run_worker(args)
    if args.command == "ui":
        return _run_ui(args)
    if args.command == "agent":
        return _run_agent(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
