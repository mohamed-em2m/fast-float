"""Inspect the resolved per-component runtime plan (and live sessions).

``python -m fast_float.runtimes inspect`` prints, for every registry
component, the backend the config resolves to, its provider/artifact, whether
the box can actually honour it, and any degradation that would silently leave
a component on eager.

``inspect_model``/``warn_degraded`` report the *live* sessions once a model has
been loaded and ``apply_backends`` has run -- used at startup so a sticky
degradation (e.g. ``fmt``) or a missing artifact cannot hide silently.
"""

from __future__ import annotations

import json
import os

from fast_float.onnx.backends import env_fingerprint, get_config, load_backends
from fast_float.onnx.registry import REGISTRY


def _compile_available() -> bool:
    try:
        from fast_float.builders.dynamo import DynamoBuilder

        return DynamoBuilder().available()
    except Exception:
        return False


def plan_rows(config=None, env=None) -> tuple[list[dict], dict]:
    """Resolved plan for every component: one dict row each, plus the env probe."""
    config = config or get_config()
    env = env if env is not None else env_fingerprint()
    rows: list[dict] = []
    for name, spec in REGISTRY.items():
        cc = config.component(name)
        mode = config.mode_for(name)
        path = cc.onnx_path or spec.onnx_default
        rows.append(
            {
                "component": name,
                "mode": mode,
                "provider": cc.provider,
                "onnx_path": path,
                "onnx_exists": bool(path) and os.path.exists(path) if mode == "onnx" else None,
                "replicable": spec.replicable,
                "degrade_on_error": spec.degrade_on_error,
                "hot": spec.hot,
                "path_source": "config" if cc.onnx_path else "default",
            }
        )
    return rows, env


def plan_warnings(rows: list[dict], env: dict, compile_available: bool | None = None) -> list[str]:
    """Human-readable reasons the resolved plan would silently degrade."""
    if compile_available is None:
        compile_available = _compile_available()
    out: list[str] = []
    for r in rows:
        name = r["component"]
        if r["mode"] == "onnx":
            if r["onnx_exists"] is False:
                out.append(f"{name}: onnx selected but artifact not found ({r['onnx_path']}); would fall back to eager")
            if r["provider"] == "trt" and not env.get("trt"):
                out.append(f"{name}: provider 'trt' selected but TensorRT is not loadable; would use CUDA/CPU")
            elif r["provider"] in (None, "cuda") and not env.get("cuda_ep"):
                out.append(f"{name}: CUDA EP not usable; would run on CPU")
        elif r["mode"] == "compile" and not compile_available:
            out.append(f"{name}: compile selected but torch.compile is unavailable; would stay eager")
    if rows and all(r["mode"] == "eager" for r in rows):
        out.append("all components eager (no acceleration configured)")
    return out


def _fmt_bool(v) -> str:
    return {True: "yes", False: "no", None: "-"}[v]


def _render(rows: list[dict], env: dict, warnings: list[str], config) -> str:
    headers = ("component", "mode", "provider", "hot", "repl", "degrade", "artifact", "exists")
    table = [
        (
            r["component"],
            r["mode"],
            r["provider"] or "-",
            _fmt_bool(r["hot"]),
            _fmt_bool(r["replicable"]),
            r["degrade_on_error"],
            r["onnx_path"] or "-",
            _fmt_bool(r["onnx_exists"]),
        )
        for r in rows
    ]
    widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(headers)]
    lines = [f"FLOAT runtimes — inspect (config: {getattr(config, 'source', '<default>')})", ""]
    lines.append("  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)))
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in table]
    lines.append("")
    keys = ("torch", "cuda", "ort", "trt", "cuda_ep", "cc")
    lines.append("environment: " + " ".join(f"{k}={env.get(k)}" for k in keys))
    if warnings:
        lines.append("")
        lines.append("warnings:")
        lines += [f"  - {w}" for w in warnings]
    return "\n".join(lines)


def inspect_main(config_path: str | None = None, as_json: bool = False) -> int:
    config = load_backends(config_path) if config_path else get_config()
    rows, env = plan_rows(config)
    warnings = plan_warnings(rows, env)
    if as_json:
        print(json.dumps({"components": rows, "environment": env, "warnings": warnings}, indent=2, default=str))
    else:
        print(_render(rows, env, warnings, config))
    return 0


def inspect_model(model) -> list[dict]:
    """Live sessions installed on a loaded model (best-effort, never raises)."""
    from fast_float.runtimes.base_runtime import RuntimeSession

    rows: list[dict] = []
    seen: set[int] = set()

    def add(owner, attr: str, sess: RuntimeSession) -> None:
        if id(sess) in seen:
            return
        seen.add(id(sess))
        try:
            providers = list(sess.providers)
        except Exception:
            providers = []
        rows.append(
            {
                "owner": type(owner).__name__,
                "attr": attr,
                "component": sess.component or attr,
                "backend": sess.backend,
                "providers": providers,
                "device_id": sess.device_id,
                "degraded": sess.degraded,
                "reason": sess.stats.reason,
                "calls": sess.stats.calls,
            }
        )

    for mod in model.modules():
        for attr, val in list(vars(mod).items()):
            if isinstance(val, RuntimeSession):
                add(mod, attr, val)
        for attr, val in list(getattr(mod, "_modules", {}).items()):
            if isinstance(val, RuntimeSession):
                add(mod, attr, val)
    return rows


def warn_degraded(model, logger) -> list[dict]:
    """Log a warning for every degraded session; returns the rows (for callers)."""
    rows = inspect_model(model)
    for r in rows:
        if r["degraded"]:
            logger.warning(
                f"backends: {r['component']} runtime degraded ({r['reason']}); "
                f"serving {r['backend']} — check the artifact/provider"
            )
    return rows


__all__ = ["inspect_main", "inspect_model", "plan_rows", "plan_warnings", "warn_degraded"]
