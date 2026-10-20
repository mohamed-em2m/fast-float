"""``python -m fast_float.runtimes inspect`` — show the resolved runtime plan."""

from __future__ import annotations

import argparse

from fast_float.runtimes import inspect as _inspect


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fast_float.runtimes", description="Inspect FLOAT component runtimes")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("inspect", help="show the resolved per-component runtime plan")
    p.add_argument("--config", default=None, help="backend config path (default: resolved config)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    if args.cmd in (None, "inspect"):
        return _inspect.inspect_main(config_path=getattr(args, "config", None), as_json=getattr(args, "json", False))
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
