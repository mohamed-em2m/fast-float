"""Unit tests for fast-float CLI argument parsing and dispatch."""

from fast_float.cli import build_parser


def test_cli_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["serve"])
    assert args.command == "serve"
    assert args.host == "0.0.0.0"
    assert args.port == 8000
    assert args.server_type == "hypercorn"
    assert args.dual_process is True
    assert not args.frontend
    assert args.frontend_port == 3000
    assert args.control_port == 5555
    assert args.frames_port == 5556


def test_cli_parser_custom_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--frontend",
            "--frontend-port",
            "3001",
            "--server-type",
            "uvicorn",
            "--ngpus",
            "2",
        ]
    )
    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 9000
    assert args.server_type == "uvicorn"
    assert args.dual_process is True  # default
    assert args.frontend is True
    assert args.frontend_port == 3001
    assert args.ngpus == 2


def test_cli_parser_no_dual_process():
    """--no-dual-process opts out of the default dual-process mode."""
    parser = build_parser()
    args = parser.parse_args(["serve", "--no-dual-process"])
    assert args.dual_process is False


def test_cli_parser_worker_subcommand():
    parser = build_parser()
    args = parser.parse_args(["worker", "--control-port", "6555", "--frames-port", "6556"])
    assert args.command == "worker"
    assert args.control_port == 6555
    assert args.frames_port == 6556


def test_cli_parser_ui_subcommand():
    parser = build_parser()
    args = parser.parse_args(["ui", "--port", "3333"])
    assert args.command == "ui"
    assert args.port == 3333
